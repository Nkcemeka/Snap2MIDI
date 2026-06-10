import math
import torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import defaultdict
from .crf import NeuralSemiCRFInterval
from .utilities import *
from .eval_utils import *
from .layersTransformer import *
from .train_utils import PairwiseFeatureBatch, MovingBuffer
import torch_optimizer as optim
import torchmetrics


class TrainTranskunMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "ngt", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "nest", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "ncorrect", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "ngt_frame", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "nest_frame", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "ncorrect_frame", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "sev", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "seof", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
    
    def update(self, ngt, nest, ncorrect, ngt_frame, nest_frame, ncorrect_frame, \
        sev, seof):
        self.ngt += ngt
        self.nest += nest
        self.ncorrect += ncorrect
        self.ngt_frame += ngt_frame
        self.nest_frame += nest_frame
        self.ncorrect_frame += ncorrect_frame
        self.sev += sev
        self.seof += seof
    
    def compute(self):
        result = {}
        total_ngt = self.ngt.item() + 1e-4
        total_nest = self.nest.item() + 1e-4
        total_ncorrect = self.ncorrect.item() + 1e-4
        precision = total_ncorrect/total_nest
        recall = total_ncorrect/total_ngt
        f1 = (2*precision*recall)/(precision+recall)

        result["train/train_f1"] = f1
        result["train/train_precision"] = precision
        result["train/train_recall"] = recall

        total_ngt_frame = self.ngt_frame.item() + 1e-4
        total_nest_frame = self.nest_frame.item() + 1e-4
        total_ncorrect_frame = self.ncorrect_frame.item() + 1e-4
        precision_frame = total_ncorrect_frame/total_nest_frame
        recall_frame = total_ncorrect_frame/total_ngt_frame
        f1_frame = (2*precision_frame*recall_frame)/(precision_frame + recall_frame)
        total_sev = self.sev.item()
        total_seof = self.seof.item()
        result["train/train_f1_frame"] = f1_frame
        result["train/train_precision_frame"] = precision_frame
        result["train/train_recall_frame"] = recall_frame
        result["train/train_mse_velocity"] = total_sev / total_ngt
        result["train/train_mse_of"] = total_seof / total_ngt
        return result

class ValTranskunMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "ngt", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "nest", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "ncorrect", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "length", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "logProb", default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
    
    def update(self, ngt, nest, ncorrect, length, log_prob):
        self.ngt += ngt
        self.nest += nest
        self.ncorrect += ncorrect
        self.length += length
        self.logProb += log_prob
    
    def compute(self):
        result = {}
        total_ngt = self.ngt.item() + 1e-4
        total_nest = self.nest.item() + 1e-4
        total_ncorrect = self.ncorrect.item() + 1e-4
        precision = total_ncorrect/total_nest
        recall = total_ncorrect/total_ngt
        total_length = self.length.item()
        total_logp = self.logProb.item()
        f1 = (2*precision*recall)/(precision+recall)

        result["val/f1"] = f1
        result["val/precision"] = precision
        result["val/recall"] = recall
        result["val/meanNLL"] = total_logp/total_length
        return result


class ModelConfig:
    def __init__(self):
        self.f_min = 30
        self.f_max = 8000
        self.n_mels = 229
        self.segmentHopSizeInSecond = 8
        self.segmentSizeInSecond = 16
        self.hopSize = 1024
        self.windowSize = 4096
        self.fs = 44100
        self.nExtraWins = 5
        self.baseSize = 40
        self.downsampleF = True
        self.posEmbedInitGamma = 1
        self.nHead = 4
        self.fourierSize = 64
        self.nLayers = 6 
        self.enabledAttn =["F", "T"]
        self.hiddenFactorAttn= 1
        self.hiddenFactor = 4
        self.velocityPredictorHiddenSize = 512
        self.refinedOFPredictorHiddenSize = 512
        self.scoringExpansionFactor = 4
        self.useInnerProductScorer = True
        self.scoreDropoutProb = 0.1
        self.contextDropoutProb = 0.1
        self.velocityDropoutProb = 0.1
        self.refinedOFDropoutProb= 0.1
        self.weight_decay = 0.0001
        self.max_lr = 0.0002
        self.nIter = 180000
        self.quantile = 0.8

    def __repr__(self):
        return repr(self.__dict__)

Config = ModelConfig

class Transkun(pl.LightningModule):
    Config = ModelConfig

    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.hopSize = conf.hopSize

        self.windowSize = conf.windowSize
        self.fs = conf.fs

        self.segmentSizeInSecond = conf.segmentSizeInSecond
        self.segmentHopSizeInSecond = conf.segmentHopSizeInSecond

        self.framewiseFeatureExtractor = MelSpectrum(conf.windowSize,
            f_min = conf.f_min, f_max = conf.f_max, n_mels = conf.n_mels, \
            fs = conf.fs, nExtraWins = conf.nExtraWins, log=True, \
            toMono=True)

        self.targetMIDIPitch= [-64, -67] + list(range(21, 108+1))

        useInnerProductScorer = conf.useInnerProductScorer
        self.useInnerProductScorer = useInnerProductScorer
        if useInnerProductScorer:
            self.scorer = ScaledInnerProductIntervalScorer(
                    conf.baseSize*conf.scoringExpansionFactor,
                    1,
                    dropoutProb = conf.scoreDropoutProb)
        else:
            self.scorerProj = nn.Linear(len(self.targetMIDIPitch)*conf.baseSize, 512)
            self.scorer = PairwiseFeatureBatch(512, outputSize= len(self.targetMIDIPitch) )

        self.velocityPredictor = nn.Sequential(
                                nn.Linear(conf.baseSize*3*conf.scoringExpansionFactor,
                                    conf.velocityPredictorHiddenSize),
                                nn.GELU(),
                                nn.Dropout(conf.velocityDropoutProb),
                                nn.Linear(conf.velocityPredictorHiddenSize, 128)
                                )

        # output dequantize offset + presence logits
        self.refinedOFPredictor = nn.Sequential(
            nn.Linear(conf.baseSize*3*conf.scoringExpansionFactor,
                                    conf.refinedOFPredictorHiddenSize),
            nn.GELU(),
            nn.Dropout(conf.refinedOFDropoutProb),
            nn.Linear(conf.refinedOFPredictorHiddenSize, 4)
            )
        
        self.backbone = Backbone(
            inputSize = self.framewiseFeatureExtractor.nChannel,
            baseSize = conf.baseSize,
            posEmbedInitGamma = conf.posEmbedInitGamma,
            nHead = conf.nHead,
            fourierSize = conf.fourierSize,
            hiddenFactor = conf.hiddenFactor,
            hiddenFactorAttn = conf.hiddenFactorAttn,
            expansionFactor = conf.scoringExpansionFactor,
            nLayers = conf.nLayers,
            dropoutProb = conf.contextDropoutProb,
            enabledAttn =conf.enabledAttn,
            downsampleF = conf.downsampleF
        )

        self.gradNormHist = MovingBuffer(initValue = 40, maxLen = 10000)
        self.train_metrics = TrainTranskunMetric()
        self.val_metrics = ValTranskunMetric()
        
    def getDevice(self):
        return next(self.parameters()).device   

    def processFramesBatch(self, framesBatch):
        
        nBatch = framesBatch.shape[0]
        # nChannel = framesBatch.shape[1]
        
        # gain normalization
        # if self.training:
        framesBatchMean = torch.mean(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatchStd = torch.std(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatch = (framesBatch - framesBatchMean)/(framesBatchStd+ 1e-8)


        featuresBatch = self.framewiseFeatureExtractor(framesBatch).contiguous()

        # print(featuresBatch.shape)
        # now with shape [nBatch, nAudioChannel, nStep, NFreq, nChannel]
        nChannel = 1
        featuresBatch  = featuresBatch.view(nBatch*nChannel, *featuresBatch.shape[-3:])
        # now with shape [nBatch* nAudioChannel, nStep, NFreq, nChannel]
        ctx = self.backbone(featuresBatch, outputIndices = torch.tensor(self.targetMIDIPitch, device = featuresBatch.device))

        # construct pairwise score matrix
        if self.useInnerProductScorer:
            S_batch, S_skip_batch = self.scorer(ctx)

            # now with shape [ nStep, nStep, nBatch, nSym ] 
        else:
            # ctx = ctx.
            ctxScore = ctx.permute(2, 0, 1,3).flatten(-2,-1)
            S_batch, S_skip_batch = self.scorer(self.scorerProj(ctxScore), 10240)

        # batch the CRF together 
        S_batch = S_batch.flatten(-2,-1)
        S_skip_batch= S_skip_batch.flatten(-2,-1)

        # with shape [*, nBatch*nSym]
        crf = NeuralSemiCRFInterval(S_batch, S_skip_batch)
        return crf, ctx

    def log_prob(self, xBatch, notesBatch):
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)
        # now with shape[nbatch, nAudioChannel, nSample]

        device = xBatch.device

        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # crf returned would be nBatch*nChannel flattened for batch processing
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)

        # prepare groundtruth 
        intervalsBatch= []
        velocityBatch = []
        ofRefinedGTBatch = []
        ofPresenceGTBatch = []
        for notes in notesBatch:
            data = prepareIntervals(notes, self.hopSize/self.fs, self.targetMIDIPitch)
            intervalsBatch .append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))
            ofPresenceGTBatch.append(sum(data["endPointPresence"], []))


        intervalsBatch_flatten = sum(intervalsBatch , [])
        assert( len(intervalsBatch_flatten) == nBatch* len(self.targetMIDIPitch))
        pathScore = crfBatch.evalPath(intervalsBatch_flatten) 
        logZ = crfBatch.computeLogZ()
        logProb = pathScore - logZ
        logProb = logProb.view(nBatch, -1)


        # then fetch the attrbute features for all intervals
        nIntervalsAll =  sum([len(_) for _ in intervalsBatch_flatten])
        if nIntervalsAll>0:
            ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)
            attributeInput = torch.cat([ctx_a_all,
                                       ctx_b_all,
                                       ctx_a_all*ctx_b_all
                                       ],dim = -1)

            
            # prepare groundtruth for velocity
            velocityBatch = sum(velocityBatch, [] )
            ofRefinedGTBatch = sum(ofRefinedGTBatch, [] )
            ofPresenceGTBatch = sum(ofPresenceGTBatch, [])

            logitsVelocity = self.velocityPredictor(attributeInput)
            logitsVelocity = F.log_softmax(logitsVelocity, dim = -1)

            velocityBatch= torch.tensor(velocityBatch, dtype =torch.long, device = device)

            logProbVelocity = torch.gather(logitsVelocity, dim = -1, index = velocityBatch.unsqueeze(-1)).squeeze(-1)

            ofRefinedGTBatch = torch.tensor(ofRefinedGTBatch, device = device, dtype = torch.float)
            ofPresenceGTBatch = torch.tensor(ofPresenceGTBatch, device = device, dtype = torch.float)
            

            # shift it to [0,1]
            # print("GT:", ofRefinedGTBatch)
            ofRefinedGTBatch = ofRefinedGTBatch*0.99+0.5
            ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)
            # ofValue = F.logsigmoid(ofValue)
            ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)
            logProbOF = ofDist.log_prob(ofRefinedGTBatch).sum(-1)
            ofPresenceDist = torch.distributions.Bernoulli(logits = ofPresence)
            logProbOFPresence = ofPresenceDist.log_prob(ofPresenceGTBatch).sum(-1)
            # scatter them back
            logProb = logProb.view(-1)
            logProb = logProb.scatter_add(-1, scatterIdx_all, logProbVelocity+ logProbOF + logProbOFPresence)

        logProb = logProb.view(nBatch, -1)

        return logProb

    def computeStatsMIREVAL(self, xBatch, notesBatch):
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)
        device = xBatch.device
        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # get the transcription from the frames
        notesEstBatch, _ = self.transcribeFrames(framesBatch)
        assert(len(notesBatch) == len(notesEstBatch))
        # metricsBatch = [Evaluation.compareTranscription(est, gt) for est, gt in zip(notesEstBatch, notesBatch)  ]
        # aggregate by  batch count
        metricsBatch = []
        nEstTotal = 0
        nGTTotal = 0 
        nCorrectTotal = 0
          
        for est, gt in zip(notesEstBatch, notesBatch):
            metrics = compareTranscription(est, gt) 
            p,r,f, _ = metrics["note+offset"]

            nGT= metrics["nGT"]
            nEst= metrics["nEst"]

            nCorrect = r*nGT

            nEstTotal+= nEst
            nGTTotal+= nGT
            nCorrectTotal += nCorrect


        stats = {
                "nGT": nGTTotal,
                "nEst": nEstTotal, 
                "nCorrect": nCorrectTotal,
                }

        return stats

    def computeStats(self, xBatch, notesBatch):
        # print(xBatch.shape)
        # print(len(notesBatch))
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)
        device = xBatch.device
        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # crf returned would be nBatch*nChannel flattened for batch processing
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)

        path = crfBatch.decode()

        # print(sum([len(p) for p in path]))
        intervalsBatch= []
        velocityBatch = []
        ofRefinedGTBatch = []
        for notes in notesBatch:
            data = prepareIntervals(notes, self.hopSize/self.fs, self.targetMIDIPitch)
            intervalsBatch .append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))

        intervalsBatch_flatten = sum(intervalsBatch , [])
        assert( len(intervalsBatch_flatten) == nBatch* len(self.targetMIDIPitch))
        # then compare intervals and path
        assert(len(path) == len(intervalsBatch_flatten))
            
        
        # print(sum([len(p) for p in intervalsBatch_flatten]), "intervalsGT")
        statsAll = [compareBracket(l1,l2) for l1, l2 in zip(path, intervalsBatch_flatten)]

        nGT = sum([_[0] for _ in statsAll])
        nEst = sum([_[1] for _ in statsAll])
        nCorrect= sum([_[2] for _ in statsAll])
 
        # omit pedal
        statsFramewiseAll = [compareFramewise(l1,l2) for l1, l2 in zip(path, intervalsBatch_flatten)]
        nGTFramewise = sum([_[0] for _ in statsFramewiseAll])
        nEstFramewise = sum([_[1] for _ in statsFramewiseAll])
        nCorrectFramewise = sum([_[2] for _ in statsFramewiseAll])

        # then make forced predictions about velocity and refined onset offset
        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   ],dim = -1)

        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)

        #MSE
        w = torch.arange(128, device = device)
        velocity = (pVelocity*w).sum(-1)
        ofValue, _ = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)


        velocityBatch = sum(velocityBatch, [] )
        velocityBatch= torch.tensor(velocityBatch, dtype =torch.long, device = device)

        ofRefinedGTBatch = sum(ofRefinedGTBatch, [] )
        ofRefinedGTBatch = torch.tensor(ofRefinedGTBatch, device = device, dtype = torch.float)
        # compare p velocity with ofValue


        # ofValue-of
        seOF = (ofValue-ofRefinedGTBatch).pow(2).sum()
        seVelocity= (velocity-velocityBatch).pow(2).sum()

        # print(ofValue[0], ofRefinedGTBatch[0])
        # print(ofValue[-1], ofRefinedGTBatch[-1])

        stats = {
                "nGT": nGT,
                "nEst": nEst, 
                "nCorrect": nCorrect,
                "nGTFramewise": nGTFramewise,
                "nEstFramewise": nEstFramewise, 
                "nCorrectFramewise": nCorrectFramewise,
                "seVelocityForced": seVelocity.item(),
                "seOFForced": seOF.item(),
                }
        return stats

    def fetchIntervalFeaturesBatch(self, ctxBatch, intervalsBatch):
        # ctx: [N, SYM, T, D]
        ctx_a_all = []
        ctx_b_all = []
        symIdx_all = []
        scatterIdx_all = []
        device = ctxBatch.device
        T = ctxBatch.shape[-2]

        for idx, curIntervals in enumerate(intervalsBatch):
            nIntervals =len(sum(curIntervals, []))
            if nIntervals>0:
                symIdx = torch.tensor(listToIdx(curIntervals), dtype=torch.long, device = device)
                symIdx_all.append(symIdx)

                scatterIdx_all.append(idx*len(self.targetMIDIPitch)+ symIdx)

                indices = torch.tensor(sum(curIntervals, []), dtype =torch.long, device = device)
                # print(len(symIdx), len(indices[:,0]))

                ctx_a = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 0]+ symIdx*T)
                ctx_b = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 1]+ symIdx*T)

                ctx_a_all.append(ctx_a)
                ctx_b_all.append(ctx_b)

        ctx_a_all = torch.cat(ctx_a_all, dim = 0)
        ctx_b_all = torch.cat(ctx_b_all, dim = 0)
        symIdx_all= torch.cat(symIdx_all, dim = 0)
        scatterIdx_all= torch.cat(scatterIdx_all, dim = 0)

        return ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all

    def transcribeFrames(self,framesBatch, forcedStartPos= None, velocityCriteron = "hamming", onsetBound = None, lastFrameIdx = None):
        device = framesBatch.device
        nBatch=  framesBatch.shape[0]
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)
        nSymbols = len(self.targetMIDIPitch)
        nFrame = framesBatch.shape[-2]

        if lastFrameIdx is None:
            lastFrameIdx = nFrame-1
        path = crfBatch.decode(forcedStartPos = forcedStartPos, forward=False)

        assert(nSymbols*nBatch == len(path))

        # also get the last position for each path for forced decoding
        if onsetBound is not None:
            path = [[e for e in _ if e[0]<onsetBound] for _ in path]
        
        # then predict attributes associated with frames
        # obtain segment features
        nIntervalsAll =  sum([len(_) for _ in path])
        # print("#e:", nIntervalsAll)
        intervalsBatch = []
        for idx in range(nBatch):
            curIntervals =  path[idx*nSymbols: (idx+1)*nSymbols]
            intervalsBatch.append(curIntervals)
       
        if nIntervalsAll == 0:
            # nothing detected, return empty
            return [[] for _ in range(nBatch)], [0 for _ in range(len(path))]

        # then predict the attribute set
        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)
        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   ],dim = -1)

        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)
        
        #MSE
        if velocityCriteron == "mse":
            w = torch.arange(128, device = device)
            velocity = (pVelocity*w).sum(-1)
        elif velocityCriteron == "match":
            #TODO: Minimal risk 
            # predict velocity, readout by minimizing the risk
            # 0.1 is usually the tolerance for the velocity, so....
            
            # It will never make so extreme predictions

            # create the risk matrix
            w = torch.arange(128, device = device)

            # [Predicted, Actual]

            tolerance = 0.1* 128
            utility = ((w.unsqueeze(1)- w.unsqueeze(0)).abs()<tolerance).float()

            r = pVelocity@utility

            velocity = torch.argmax(r, dim = -1)
        elif velocityCriteron == "hamming":
            # return the mode
            velocity = torch.argmax(pVelocity, dim = -1)

        elif velocityCriteron == "mae":
            # then this time return the median
            pCum = pVelocity.cumsum(-1)
            tmp = (pCum-0.5)>0
            w2 = torch.arange(128, 0. ,-1, device = device)

            velocity = torch.argmax(tmp*w2, dim = -1)
        else:
            raise Exception("Unrecognized criterion: {}".format(velocityCriteron))
        ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        ofPresence = ofPresence>0

        # generate the final result
        # parse the list of path to (begin, end, midipitch, velocity) 
        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()
        ofPresence = ofPresence.cpu().detach().tolist()

        assert(len(velocity) == len(ofValue))
        assert(len(velocity) == nIntervalsAll)
         

        nCount = 0 
        notes = [[] for _ in range(nBatch)]
        frameDur = self.hopSize/self.fs
        # the last offset
        lastP = []

        for idx in range(nBatch):
            curIntervals = intervalsBatch[idx]
            for j, eventType in enumerate(self.targetMIDIPitch):
                lastEnd = 0
                curLastP = 0

                for k, aInterval in enumerate(curIntervals[j]):
                    # print(aInterval, eventType, velocity[nCount], ofValue[nCount])
                    isLast = (k == (len(curIntervals[j])-1) )
                    
                    curVelocity = velocity[nCount]

                    curOffset = ofValue[nCount]
                    start = (aInterval[0]+ curOffset[0] )*frameDur
                    end = (aInterval[1]+ curOffset[1])*frameDur

                    # ofPresence prediction is only used to distinguish the corner case that either onset or offset happens exactly on the first/last frame.

                    hasOnset = (aInterval[0]>0) or ofPresence[nCount][0]
                    hasOffset = (aInterval[1]<lastFrameIdx) or ofPresence[nCount][1]

                    assert(aInterval[0]>= 0)
                    # print(aInterval[0], aInterval[1], nFrame)
                    start = max(start, lastEnd)
                    end = max(end, start+1e-8)
                    lastEnd = end
                    curNote = Note(
                         start = start,
                         end = end,
                         pitch = eventType,
                         velocity = curVelocity,
                         hasOnset = hasOnset,
                         hasOffset = hasOffset)
                    
                    notes[idx].append(curNote)

                    if hasOffset:
                        curLastP = aInterval[1]
                    # if hasOnset and hasOffset:
                        # curLastP = aInterval[1]


                    nCount+= 1
                lastP.append(curLastP)
            notes[idx].sort(key = lambda x: (x.start, x.end,x.pitch))
        return notes, lastP

    def transcribe(self, x, stepInSecond = None, segmentSizeInSecond = None, discardSecondHalf=False, mergeIncompleteEvent = True):
        if stepInSecond is None and segmentSizeInSecond is None:
            stepInSecond = self.segmentHopSizeInSecond
            segmentSizeInSecond = self.segmentSizeInSecond

        x= x.transpose(-1,-2)

        # gain normalization
        # x = (x-x.mean())/(x.std()+1e-8)

        padTimeBegin = (segmentSizeInSecond-stepInSecond)
        x = F.pad(x, (math.ceil(padTimeBegin*self.fs), math.ceil(self.fs* (padTimeBegin))))
        nSample = x.shape[-1]
        eventsAll= []
        eventsByType= defaultdict(list)
        startFrameIdx = math.floor(padTimeBegin*self.fs/self.hopSize)
        startPos = [startFrameIdx]* len(self.targetMIDIPitch)
        # startPos =None

        stepSize = math.ceil(stepInSecond*self.fs/self.hopSize)*self.hopSize
        segmentSize = math.ceil(segmentSizeInSecond*self.fs)

        for i in range(0, nSample, stepSize):
            # t1 = time.time()
            j = min(i+ segmentSize, nSample)
            # print(i, j)
            beginTime = (i)/ self.fs -padTimeBegin
            # print(beginTime)
            curSlice = x[:, i:j]
            if curSlice.shape[-1]< segmentSize:
                # pad to the segmentSize
                curSlice = F.pad(curSlice, (0, segmentSize- curSlice.shape[-1]))

            curFrames = makeFrame(curSlice, self.hopSize, self.windowSize)
            lastFrameIdx = round(segmentSize/self.hopSize)
            # # print(curSlice.shape)
            # # print(startPos)
            # startPos = None
            if discardSecondHalf:
                onsetBound = stepSize
            else:
                onsetBound = None

            curEvents, lastP = self.transcribeFrames(curFrames.unsqueeze(0), forcedStartPos = startPos, velocityCriteron = "hamming", onsetBound= onsetBound, lastFrameIdx = lastFrameIdx)
            curEvents = curEvents[0]


            startPos = []
            for k in lastP:
                startPos.append(max(k-int(stepSize/self.hopSize), 0))

            # # shift all notes by beginTime
            for e in  curEvents:
                e.start += beginTime
                e.end  += beginTime 

                e.start = max(e.start, 0)
                e.end = max(e.end, e.start)
                # print(e.start, e.end, e.pitch, e.hasOnset, e.hasOffset)

            for e in curEvents:
                if mergeIncompleteEvent:
                    if len(eventsByType[e.pitch])>0:
                        last_e = eventsByType[e.pitch][-1]

                        # test if e overlap with the last event 
                        if e.start < last_e.end:
                            if e.hasOnset: #and e.hasOffset:
                                eventsByType[e.pitch][-1] = e
                            else:
                                # merge two events
                                eventsByType[e.pitch][-1].hasOffset = e.hasOffset
                                eventsByType[e.pitch][-1].end = max(e.end, last_e.end)
                                # eventsByType[e.pitch][-1].end = max(e.end, last_e.end)
                            continue
                if e.hasOnset:
                    eventsByType[e.pitch].append(e)
            eventsAll.extend(curEvents)

        # handling incomplete events in the last segment
        for eventType in eventsByType:
            if len(eventsByType[eventType])>0:
                eventsByType[eventType][-1].hasOffset = True

        # flatten all events
        eventsAll = sum(eventsByType.values(), [])

        # post filtering
        eventsAll = [n for n in eventsAll if n.hasOffset]
        eventsAll = resolveOverlapping(eventsAll)
        return eventsAll
    
    def getOptimizerGroup(self):
        # param_optimizer = list(self.named_parameters())
        noDecay = []
        for name, module in self.named_modules():
            if isinstance(module, nn.GroupNorm) \
                    or isinstance(module, nn.LayerNorm) \
                    or isinstance(module, LearnableSpatialPositionEmbedding):
                noDecay.extend(list(module.parameters()))
            else:
                noDecay.extend([p for n, p in module.named_parameters() if "bias" in n])

        otherParams  =set(self.parameters()) - set(noDecay)
        otherParams = [param for param in self.parameters() if param in otherParams]
        noDecay = set(noDecay)
        noDecay = [param for param in self.parameters() if param in noDecay]

        optimizerConfig = [{"params": otherParams},
                            {"params": noDecay, "weight_decay":0e-7}]
        return optimizerConfig

    def training_step(self, batch, batch_idx):
        audio = batch["audioSlices"]
        notes = batch["notes"]
        audio_length = (
                audio.shape[1]
                / self.conf.fs
        )
        logp = self.log_prob(audio, notes)
        loss = (-logp.sum(-1).mean())

        self.log_dict({
            "train_loss": (loss/audio_length).item()
        }, logger=True, on_step=True, sync_dist=True)


        # Log the learning rate so we can track its behaviour
        optimizer = self.optimizers()
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )
        self.log(
            "optimizer/lr",
            current_lr,
            on_step=True,
            logger=True,
            sync_dist=False,
        )
        return loss/50

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 40 == 0:
            audio = batch["audioSlices"]
            notes = batch["notes"]
            with torch.no_grad():
                self.eval()
                stats = self.computeStats(audio, notes)
                stats2 = self.computeStatsMIREVAL(audio, notes)
            
            Gt = stats2["nGT"]
            Est = stats2["nEst"]
            Correct = stats2["nCorrect"]
            GTFramewise = stats["nGTFramewise"]
            EstFramewise = stats["nEstFramewise"]
            CorrectFramewise = stats["nCorrectFramewise"]
            SEVelocity = stats["seVelocityForced"]
            SEof = stats["seOFForced"]
            self.train_metrics.update(
                Gt, Est, Correct, GTFramewise, EstFramewise, CorrectFramewise, \
                SEVelocity, SEof
            )

            results = self.train_metrics.compute()
            self.log_dict(
                results,
                logger=True,
                sync_dist=True
            )
            self.train_metrics.reset()
            self.train()
    
    def configure_optimizers(self):
        optimizer = optim.AdaBelief(
            self.getOptimizerGroup(),
            lr=1e-5, #self.conf.max_lr,
            weight_decouple=True,
            weight_decay=1e-2,#self.conf.weight_decay,
            eps=1e-8,
            rectify=True
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=4e-4,#self.conf.max_lr,
            total_steps=500000,#self.conf.nIter,
            pct_start=0.05,
            cycle_momentum=False,
            final_div_factor=2,
            div_factor=20
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        curClipValue = self.gradNormHist.getQuantile(self.conf.quantile)
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=curClipValue
        )
        self.gradNormHist.step(total_norm.item())
        self.log("optimizer/clipValue", curClipValue, logger=True,\
                  on_step=True, sync_dist=False)
        self.log("optimizer/gradNorm", total_norm.item(), logger=True,\
                  on_step=True, sync_dist=False)

    def validation_step(self, batch, batch_idx):
        audio = batch["audioSlices"]
        notes = batch["notes"]
        logp = self.log_prob(audio, notes)
        loss = -logp.sum(-1).mean()
        stats = self.computeStatsMIREVAL(audio, notes)
        Gt = stats["nGT"]
        Est = stats["nEst"]
        Correct = stats["nCorrect"]
        length = audio.shape[1]/self.conf.fs
        self.val_metrics.update(Gt, Est, Correct, length, loss.item())
    
    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.log_dict(
            results,
            logger=True,
            sync_dist=True,
            on_epoch=True
        )

