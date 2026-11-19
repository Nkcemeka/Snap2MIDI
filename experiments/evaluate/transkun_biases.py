"""TransKun transcribed at several note biases from one forward pass.

The note bias is a constant added to every interval score of TransKun's
semi-CRF before Viterbi decoding (models.transkun). transcribe() decodes
overlapping segments and carries state between them (each segment's forced
start positions come from the previous segment's path), so each bias needs
its own chain: transcribe_all_biases is TransKun's segment loop
(ModelTransformer.transcribe, discardSecondHalf=False) with one stitching
state per bias, the network run once per segment and the score matrix tiled
once per bias in the CRF's batch dimension. decode_batched is the stock
viterbiBackward with its backtrack vectorised over the columns, the same walk
step for step. The batched decoder is installed only for the duration of a
call. Used by models.transkun_grid for selection grids
(experiments/select/grid.py).

Adapted from TransKun (https://github.com/Yujia-Yan/Transkun, transkun/
ModelTransformer.py and transkun/CRF at commit c5cb837), MIT License,
Copyright (c) 2021 Yujia Yan. TransKun must be importable (models._transkun_model
puts transkun_repo on sys.path).
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transkun import Data
from transkun.CRF.NeuralSemiCRFInterval import NeuralSemiCRFInterval, viterbi
from transkun.Util import makeFrame


@torch.jit.script
def viterbi_forward(score, noiseScore):
    """viterbiBackward's dynamic programme, verbatim, returning what its
    backtrack consumes: the pointer table, the diagonal inclusions and whether
    the final frame is a singleton."""
    T = score.shape[0]
    nBatch = score.shape[2]
    q = torch.zeros(T, nBatch, device=score.device)
    ptr = []
    scoreT = score.transpose(0, 1).contiguous()
    q[T - 1] = score[T - 1, T - 1, :] * (score[T - 1, T - 1, :] > 0)
    for i in range(1, T):
        subScore = scoreT[T - i - 1, T - i:, :]
        tmp = torch.cat([q[T - i:T - i + 1, :] + noiseScore[T - i - 1, :],
                         q[T - i:, :] + subScore], dim=0)
        curV, selection = tmp.max(dim=0)
        ptr.append(selection - 1)
        singletonMask = score[T - i - 1, T - i - 1, :] > 0
        q[T - i - 1] = curV + score[T - i - 1, T - i - 1, :] * singletonMask
    ptrT = torch.stack(ptr, dim=0).cpu()
    diag = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu()
    last = (score[T - 1, T - 1, :] > 0).cpu()
    return ptrT, diag, last


def decode_batched(self, forcedStartPos=None, forward=False):
    """NeuralSemiCRFInterval.decode with the backtrack vectorised over columns.

    viterbiBackward walks each column with `while j < T-1`: a singleton (j, j)
    when the diagonal is positive, then either j += 1 (skip) or an interval
    (j, i) with i = ptr + j + 1 and j = i. Every column here advances one such
    step per iteration; j is strictly increasing within a column, so sorting
    the collected pairs by (column, j, singleton-first) restores the walk order.
    """
    if forward:
        return viterbi(self.score, self.noiseScore, forcedStartPos)
    ptr, diag, last = viterbi_forward(self.score, self.noiseScore)
    ptr, diag, last = ptr.numpy(), diag.numpy(), last.numpy()
    T, nBatch = self.score.shape[0], self.score.shape[2]
    j = np.zeros(nBatch, dtype=np.int64) if forcedStartPos is None \
        else np.asarray(forcedStartPos, dtype=np.int64)
    cols = np.arange(nBatch)
    col_out, j_out, i_out, kind_out = [], [], [], []
    active = j < T - 1
    while active.any():
        c, jj = cols[active], j[active]
        sel = ptr[T - jj - 2, c]
        hit = diag[c, jj]
        col_out.append(c[hit]); j_out.append(jj[hit]); i_out.append(jj[hit])
        kind_out.append(np.zeros(int(hit.sum()), dtype=np.int64))
        iv = sel >= 0
        i = sel[iv] + jj[iv] + 1
        col_out.append(c[iv]); j_out.append(jj[iv]); i_out.append(i)
        kind_out.append(np.ones(int(iv.sum()), dtype=np.int64))
        nj = jj + 1
        nj[iv] = i
        j[active] = nj
        active = j < T - 1
    c = np.concatenate(col_out); jv = np.concatenate(j_out)
    iv_ = np.concatenate(i_out); k = np.concatenate(kind_out)
    order = np.lexsort((k, jv, c))
    c, jv, iv_ = c[order], jv[order], iv_[order]
    result = [[] for _ in range(nBatch)]
    for cc, a, b in zip(c.tolist(), jv.tolist(), iv_.tolist()):
        result[cc].append((a, b))
    for cc in np.nonzero(last)[0].tolist():
        result[cc].append((T - 1, T - 1))
    return result



def transcribe_all_biases(model, audio, biases):
    """{bias: notes}: TransKun.transcribe() with one stitching chain per bias.

    Line for line the stock segment loop (ModelTransformer.transcribe, defaults,
    discardSecondHalf=False, mergeIncompleteEvent=True), except that the
    segment is decoded as a batch of len(biases) elements. processFramesBatch is
    replaced for the call so the network runs on one copy of the frames and
    the CRF it returns holds every bias's columns.
    """
    nb, nSym = len(biases), len(model.targetMIDIPitch)
    device = audio.device
    bias_col = torch.tensor(biases, dtype=torch.float32, device=device).repeat_interleave(nSym)
    original = model.processFramesBatch

    def tiled(frames):
        crf, ctx = original(frames[:1])
        # columns are batch*nSym + sym, as S_batch.flatten(-2, -1) lays them out
        score = crf.score.repeat(1, 1, nb) + bias_col
        noise = crf.noiseScore.repeat(1, nb)
        return NeuralSemiCRFInterval(score, noise), ctx.expand(nb, *ctx.shape[1:])

    fs, hop = model.fs, model.hopSize
    stepInSecond, segmentSizeInSecond = model.segmentHopSizeInSecond, model.segmentSizeInSecond
    x = audio.transpose(-1, -2)
    padTimeBegin = segmentSizeInSecond - stepInSecond
    x = F.pad(x, (math.ceil(padTimeBegin * fs), math.ceil(fs * padTimeBegin)))
    nSample = x.shape[-1]
    startFrameIdx = math.floor(padTimeBegin * fs / hop)
    startPos = [startFrameIdx] * (nSym * nb)
    stepSize = math.ceil(stepInSecond * fs / hop) * hop
    segmentSize = math.ceil(segmentSizeInSecond * fs)
    eventsByType = [defaultdict(list) for _ in biases]

    model.processFramesBatch = tiled
    stock_decode = NeuralSemiCRFInterval.decode
    NeuralSemiCRFInterval.decode = decode_batched
    try:
        for i in range(0, nSample, stepSize):
            j = min(i + segmentSize, nSample)
            beginTime = i / fs - padTimeBegin
            curSlice = x[:, i:j]
            if curSlice.shape[-1] < segmentSize:
                curSlice = F.pad(curSlice, (0, segmentSize - curSlice.shape[-1]))
            curFrames = makeFrame(curSlice, hop, model.windowSize)
            lastFrameIdx = round(segmentSize / hop)
            frames = curFrames.unsqueeze(0).expand(nb, *curFrames.shape)
            curEvents, lastP = model.transcribeFrames(
                frames, forcedStartPos=startPos, velocityCriteron="hamming",
                onsetBound=None, lastFrameIdx=lastFrameIdx)
            startPos = [max(k - int(stepSize / hop), 0) for k in lastP]
            for b, evs in enumerate(curEvents):
                byType = eventsByType[b]
                for e in evs:
                    e.start += beginTime
                    e.end += beginTime
                    e.start = max(e.start, 0)
                    e.end = max(e.end, e.start)
                for e in evs:
                    if len(byType[e.pitch]) > 0:
                        last_e = byType[e.pitch][-1]
                        if e.start < last_e.end:
                            if e.hasOnset:
                                byType[e.pitch][-1] = e
                            else:
                                byType[e.pitch][-1].hasOffset = e.hasOffset
                                byType[e.pitch][-1].end = max(e.end, last_e.end)
                            continue
                    if e.hasOnset:
                        byType[e.pitch].append(e)
    finally:
        del model.processFramesBatch      # back to the class method
        NeuralSemiCRFInterval.decode = stock_decode

    out = {}
    for b, byType in zip(biases, eventsByType):
        for evs in byType.values():
            if len(evs) > 0:
                evs[-1].hasOffset = True
        eventsAll = [n for n in sum(byType.values(), []) if n.hasOffset]
        out[b] = Data.resolveOverlapping(eventsAll)
    return out
