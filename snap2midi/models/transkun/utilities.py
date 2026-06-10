import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict
from ncls import FNCLS

def computeParamSize(module):
    total_params = sum(p.numel() for p in module.parameters())
    # Convert to millions
    total_params_millions = total_params / 1e6
    return total_params_millions

def checkpointByPass(f, *args):
    return f(*args)

def checkpointSequentialByPass(f,n, *args):
    return f(*args)

def makeFrame(x, hopSize, windowSize, leftPaddingHalfFrame =True):
    assert(hopSize<windowSize)

    nFrame = math.ceil((x.shape[-1])/hopSize)+1

    if leftPaddingHalfFrame:
        lPad = windowSize//2
        rPad = (nFrame-1)*hopSize+windowSize//2 - x.shape[-1]
    else:
        lPad = 0
        rPad = (nFrame-1)*hopSize+windowSize - x.shape[-1]

    x = F.pad(x, (lPad, rPad))
    frames = x.unfold(-1, windowSize, hopSize)
    assert(frames.shape[-2] == nFrame), (frames.shape[-2], nFrame)
    return frames

class GaussianWindows(nn.Module):
    def __init__(self, n, nWin):
        super().__init__()
        self.n = n
        self.nWin=nWin

        self.sigma = nn.Parameter(
                -torch.ones(n)*1
                )

        self.center = nn.Parameter(
                torch.Tensor(
                    torch.logit((torch.arange(1, n+1))/(n+1))
                    ))
    
    def get(self):
        sigma = torch.sigmoid(self.sigma)
        center = torch.sigmoid(self.center)

        device = next(self.parameters()).device

        x= torch.arange(self.nWin, device =device)
        Y = (-0.5* ((x.unsqueeze(1)- self.nWin*center)/(sigma*self.nWin/2))**2  ).exp()

        return Y

class Spectrum(nn.Module):
    def __init__(self, windowSize, nExtraWins = 0,  log=False):
        super().__init__()


        self.outputDim = windowSize//2+1

        self.nChannel = (nExtraWins+1)

        self.log = log
        
        # learnable window function
        self.register_buffer( "win", torch.hann_window(windowSize))

        if nExtraWins> 0:
            self.winGen = GaussianWindows(nExtraWins, windowSize)


        self.nExtraWins = nExtraWins

    def forward(self, frames):

        if self.nExtraWins>0:
            wins = self.winGen.get()
            wins = torch.cat([self.win.unsqueeze(0), wins.t()], dim = 0)
        else:
            # wins = self.win.unsqueeze(0)
            wins = torch.cat([self.win.unsqueeze(0)], dim = 0)

        spectrogram = torch.fft.rfft(
                # torch.fft.fftshift(frames*self.win),
                (frames.unsqueeze(-2)*wins),
                norm= "ortho")

        # spectrogram = spectrogram.transpose(-1,-2)
        if self.log:
            spectrogram = torch.complex(spectrogram.abs(),  spectrogram.angle())

        result = spectrogram
        result = result.transpose(-1,-2) 
        return result

class MelSpectrum(nn.Module):
    def __init__(self, windowSize, f_min, f_max, n_mels, fs, nExtraWins=0, log=False, eps = 1e-5, toMono=False):
        super().__init__()


        self.outputDim = n_mels
        self.nChannel = (nExtraWins+1)

        import torchaudio
        self.register_buffer("freq2mels", torchaudio.functional.melscale_fbanks( 
                                                n_freqs = windowSize//2+1, 
                                                f_min = f_min,
                                                f_max = f_max,
                                                n_mels = n_mels,
                                                sample_rate = fs,
                                                ))
        self.log = log 
        self.eps = eps
        self.spectrogramExtractor = Spectrum(windowSize, nExtraWins)
        self.toMono = toMono

    def forward(self, frames):
        # output format: (.,  #frame, #freqBin, #featureChannel)

        spectrogram = self.spectrogramExtractor(frames)

        spectrogram = (spectrogram).abs().pow(2)

        if self.toMono and len(spectrogram.shape)>=4:
            spectrogram = spectrogram.mean(dim = -4, keepdim = True)


        mel = (spectrogram.transpose(-1,-2)@ self.freq2mels).transpose(-1,-2)

        if self.log:
            # with normalization
            eps = self.eps
            mel = ((mel+eps).log()-math.log(eps))/(-math.log(eps))
        return mel

def listToIdx(l):
    batchIndices = [ idx for idx, curList in enumerate(l) for _ in curList]  

    return batchIndices




#### Some data utilities
class Note:
    def __init__(self, start, end, pitch, velocity, hasOnset=True, hasOffset=True):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.hasOnset = hasOnset 
        self.hasOffset = hasOffset

    def __repr__(self):
        return str(self.__dict__)


def resolveOverlapping(note_events):
    note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

    ex_note_events = []

    idx = 0     

    buffer_dict = {}
    

    
    for note_event  in note_events:

        midi_note = note_event.pitch
        # note_event.end = max(note_event.start+1e-5, note_event.end)
        # note_event.end = max(note_event.start+1e-5, note_event.end)

        if midi_note in buffer_dict.keys():
            _idx = buffer_dict[midi_note]
            if ex_note_events[_idx].end > note_event.start:
                ex_note_events[_idx].end = note_event.start


        
        buffer_dict[midi_note] = idx
        idx += 1

        ex_note_events.append(note_event)

    ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

    # else:
        # print("overlappingOnsetOffset", note_event)

    # remove all notes that has start == end
    n1 = len(ex_note_events)
    error_notes = [n for n in ex_note_events if not n.start<n.end]
    ex_note_events = [n for n in ex_note_events if n.start<n.end]
    n2 = len(ex_note_events)
    if n1!=n2:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(error_notes)


    validateNotes(ex_note_events)
    return ex_note_events


def validateNotes(notes):
    pitches = defaultdict(list)
    for n in notes:
        if len(pitches[n.pitch])>0:
            nPrev = pitches[n.pitch][-1]
            assert n.start >= nPrev.end, str(n)+ str(nPrev)
        
        assert n.start < n.end, n

        pitches[n.pitch].append(n)

def createIndexEvents(eventList):
    # internally uses ncls package
    starts = np.array([_.start for _ in eventList])
    ends = np.array([_.end for _ in eventList])

    index = FNCLS(starts, ends, np.arange(len(eventList)))

    return index


def querySingleInterval(start, end, index):
    starts = np.array([start], dtype = np.double)
    ends = np.array([end], dtype = np.double)
    queryIds = np.array([0])
    r_id, r_loc = index.all_overlaps_both(starts, ends, queryIds)

    return r_loc

def prepareIntervalsNoQuantize(notes, targetPitch):
    validateNotes(notes)
    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)


     
    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []
    
    for p in targetPitch:
        intervals = []
        endPointRefine = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert(n.start>=0),n.start
            assert(n.end>=0),n.end




            curVelocity = n.velocity


            tmp = ( n.start , n.end)

            intervals.append(tmp)
            endPointRefine.append((0, 0) )
            velocity.append(curVelocity)
                

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)
        
        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        velocity_all.append(velocity)

        
    result = {"intervals": intervals_all, "endPointRefine": endPointRefine_all, "velocity": velocity_all}
    return result

def prepareIntervals(notes, hopSizeInSecond, targetPitch):
    validateNotes(notes)
    # print("hopSizeInSecond:", hopSizeInSecond)


    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)


     
    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []
    endPointPresence_all = []
    
    
    for p in targetPitch:
        intervals = []
        endPointRefine = []
        endPointPresence = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert(n.start>=0),n.start
            assert(n.end>=0),n.end

            start_quantized = int(round(n.start/hopSizeInSecond))
            end_quantized = int(round(n.end/hopSizeInSecond))


            start_refine = n.start/hopSizeInSecond - start_quantized
            end_refine = n.end/hopSizeInSecond - end_quantized

            curVelocity = n.velocity


            tmp = ( start_quantized, end_quantized)
            tmpPresence = (n.hasOnset, n.hasOffset)
            # print(n)

            # check if two consecutive notes can be seaprated by interval representation
            if len(intervals)>0 and (start_quantized< intervals[-1][1] or (end_quantized == intervals[-1][1] and intervals[-1][0] == start_quantized) ):
                # raise Exception("two notes quantized in the same frame that cannot be separated: {}, {}".format(tmp, intervals[-1]))
                print("two notes quantized in the same frame that cannot be separated or they are overlapping: {}, {}. These two notes are merged".format(tmp, intervals[-1]))

                # asd
                # print(n)
                # print(intervals[-1])
                # print(start_quantized, end_quantized)

                # two consecutive note on event, treat as the same note, use the same velocity
                intervals[-1] = (intervals[-1][0], end_quantized)
                endPointRefine[-1] = (endPointRefine[-1][0], end_refine)
                endPointPresence[-1] = (endPointPresence[-1][0], n.hasOffset)
            else:
                intervals.append(tmp)
                endPointRefine.append((start_refine, end_refine) )
                endPointPresence.append(tmpPresence)
                velocity.append(curVelocity)
                

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)
        
        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        endPointPresence_all.append(endPointPresence)
        velocity_all.append(velocity)

    result = {"intervals": intervals_all,
              "endPointRefine": endPointRefine_all, 
              "endPointPresence": endPointPresence_all, 
              "velocity": velocity_all}
    return result

def collate_fn_batching(batch):
    notesBatch = [sample["notes"] for sample in batch]
    audioSlices = [torch.from_numpy(sample["audioSlice"]) for sample in batch]

    nAudioSamplesMin = min( [_.shape[0] for _ in audioSlices])
    nAudioSamplesMax = max( [_.shape[0] for _ in audioSlices])

    assert nAudioSamplesMax-nAudioSamplesMin < 2
    
    audioSlices = [_[:nAudioSamplesMin] for _ in audioSlices]

    audioSlices = torch.stack(
            audioSlices, dim = 0)

    return {"notes": notesBatch, "audioSlices":audioSlices} 
