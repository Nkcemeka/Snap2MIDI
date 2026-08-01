import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import pickle
import pretty_midi
import math
import random
from .utilities import Note, querySingleInterval, createIndexEvents
pretty_midi.pretty_midi.MAX_TICK = 1e10
from tqdm import tqdm
from scipy.io import wavfile

class TranskunDataset(Dataset):
    def __init__(self, emb_path: str, sample_rate: float, hopSizeInSecond: float, \
        chunkSizeInSecond: float, audioNormalize: bool=True, notesStrictlyContained: bool=True, \
        ditheringFrames: bool=True, augmentator=None):
        """ 
            Instantiate TranskunDataset class.

            Args
            ----
                emb_path (str): Path to embeddings
                sample_rate (float): Sample rate to use
                hopSizeInSecond (float): Hop size in seconds
                chunkSizeInSecond (float): Chunk size in seconds
                audioNormalize (bool): Normalize audio to [-1, 1]
                notesStrictlyContained (bool): Select only notes strictly contained
                                               in segment of interest.
                ditheringFrames (bool): Dither frames
                augmentator: Augmentator object
        """
        super().__init__()
        self.data = [] # path to npz files
        assert Path(emb_path).exists(), f"{emb_path} does not exist."
        self.data.extend(sorted(Path(emb_path).glob("*.pt")))
        self.sample_rate = sample_rate

        # load the data
        self.loaded_data = []
        self.audio_dict = {}
        # self.loaded_duration = []
        # self.loaded_indices = [] # to speed up computation
        print(f"Loading data..")
        for i, d in tqdm(enumerate(self.data)):
            with open(str(d), "rb") as f:
                obj = pickle.load(f)
            # npz = np.load(d, allow_pickle=True)
            # self.loaded_data.append(npz)
            index = createIndexEvents(obj["notes"])
            obj["index"] = index
            self.loaded_data.append(obj)
            fs, aud = self.load_audio(obj["audio_filename"])
            self.audio_dict[obj["audio_filename"]] = (fs, aud)
            # self.loaded_duration.append(npz["duration"].item())
            # self.loaded_indices.append(createIndexEvents(npz["notes"]))
        
        self.hopSizeInSecond = hopSizeInSecond
        self.chunkSizeInSecond = chunkSizeInSecond
        self.audioNormalize = audioNormalize
        self.notesStrictlyContained = notesStrictlyContained
        self.ditheringFrames = ditheringFrames
        self.augmentator = augmentator
        self.chunksAll = []
        self.epoch = 0
    
    def load_audio(self, audioPath):
        fs, data = wavfile.read(audioPath, mmap = True)
        return fs, data

    def _check_stereo_safe(self, audioSlice) -> None:
        """Reject reverb_level="peak" on multi-channel audio.

        The reverb stage under "peak" normalizes its output to a peak of 0.5,
        and it does so per call -- which here means per channel. Two channels
        then receive unrelated gains and the left/right balance is re-rolled at
        random on every excerpt the stage fires on, which is about half of them.
        Measured on a MAESTRO chunk that moves the balance by 0.76 dB on average
        and up to 2.5 dB, against 0.17 dB under "rms".

        "rms" restores each channel to its own original energy, so the stage
        leaves the balance where it found it; what movement remains comes from
        the equalizer applying one filter to two different channel spectra,
        which is what a room does and not an artifact.

        Checked once, on the first augmented item, because the channel count is
        a property of the store rather than of the excerpt.
        """
        if getattr(self, "_stereo_checked", False):
            return
        self._stereo_checked = True
        if audioSlice.shape[-1] > 1 and getattr(self.augmentator, "reverb_level", "peak") == "peak":
            raise ValueError(
                f"this store is {audioSlice.shape[-1]}-channel and the "
                "augmentator uses reverb_level='peak', which normalizes each "
                "channel separately and so randomizes the stereo balance on "
                "every excerpt the reverb fires on. Pass reverb_level='rms' "
                "for multi-channel audio.")
    
    def build_chunks(self, seed: float, epoch: int = 0):
        # `epoch` is only carried so the augmentator can vary its draw from one
        # epoch to the next. build_chunks is the right place to take it: it is
        # already called once per epoch, and the dataloader is rebuilt straight
        # after (reload_dataloaders_every_n_epochs=1), so workers fork from a
        # dataset that already holds the new value.
        print("Building chunks...")
        self.epoch = epoch
        randGen = random.Random(
            seed
        )
        chunksAll = []
        for idx, each in tqdm(enumerate(self.loaded_data)):
            duration = each["duration"].item()
            # split the duration into equal size chunks
            # add 1 more for safe guarding the boundary
            nChunks = math.ceil((duration+self.chunkSizeInSecond)/self.hopSizeInSecond)
            hopPerChunk = math.ceil(self.chunkSizeInSecond/self.hopSizeInSecond)
            for j in range(-hopPerChunk, nChunks+hopPerChunk):
                if self.ditheringFrames:
                    shift = randGen.random()-0.5
                else:
                    shift = 0
                begin = (j+ shift)*self.hopSizeInSecond - self.chunkSizeInSecond/2
                end = begin+self.chunkSizeInSecond

                # add empty frames
                if begin<duration and end > 0:
                    chunksAll.append((idx, begin, end))
        randGen.shuffle(chunksAll)
        self.chunksAll = chunksAll
    
    def __len__(self):
        return len(self.chunksAll)

    def __getitem__(self, idx):
        if idx>self.__len__():
            raise IndexError()
        
        piece_idx, begin, end = self.chunksAll[idx]

        notes, audioSlice, fs = (
            self.fetchData(
                piece_idx,
                begin,
                end,
                audioNormalize=self.audioNormalize,
                notesStrictlyContained=self.notesStrictlyContained,
            )
        )

        if self.augmentator is not None:
            # readSlice hands back (samples, channels). The Augmentator takes a
            # bare mono waveform, so each channel goes through separately and
            # the channel axis is restored afterwards.
            #
            # Every channel is passed the identical (track_id, excerpt_start,
            # epoch), so all of them draw the same room, the same equalizer
            # curve and the same pitch shift -- one microphone pair in one
            # space, not a different space per channel. Seeding from the chunk's
            # identity rather than from call order is also what lets a different
            # architecture, with a different batch size and step count, receive
            # the identical augmentation of this same chunk. Passing nothing
            # here would silently fall back to the ambient RNG and give both
            # properties up.
            self._check_stereo_safe(audioSlice)
            audioSlice = np.stack(
                [
                    self.augmentator(
                        audioSlice[:, channel],
                        track_id=self.loaded_data[piece_idx]["audio_filename"],
                        excerpt_start=begin,
                        epoch=self.epoch,
                    )
                    for channel in range(audioSlice.shape[-1])
                ],
                axis=-1,
            )

        return {
            "notes": notes,
            "audioSlice": audioSlice,
            "fs": fs,
            "begin": begin,
        }
    
    def fetchData(self, idx, begin, end, audioNormalize, notesStrictlyContained): 
        obj = self.loaded_data[idx]
        
        # fetch the notes in this interval
        if end <0 and begin<0:
            noteIndices = []
        else:
            noteIndices = querySingleInterval(max(begin,0.0), max(end, 0.0), obj["index"])
 
        #notes = [e["notes"][int(_)] for _ in noteIndices]
        notes = [obj["notes"][int(_)] for _ in noteIndices]

        # for handling notes that goes beyond the current window
        if notesStrictlyContained:
            # notes = [_ for _ in notes if _.start>= begin and _.end<end]
            notes = [Note( _.start-begin,
                           _.end-begin,
                           _.pitch, _.velocity)
                           for _ in notes if _.start>=begin and _.end<end]
            
        else:
            # trim the notes by the boudnary, notes overlapping between segments will be merged during inference
            notes = [Note(max(_.start,begin) - begin,
                          min(_.end ,end) - begin, 
                          _.pitch, 
                          _.velocity,
                          _.start>=begin,
                          _.end<end)  for _ in notes]

        # fetch the corresponding audio chunk from the file
        audio_filename = obj["audio_filename"]

        audioSlice, fs = self.readSlice(audio_filename, begin, end, self.sample_rate, audioNormalize)
        return notes, audioSlice, fs
    
    def readSlice(self, audioPath, begin, end, fs: float, normalize=True):
        """ 
            read audio based on [begin, end]

            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            ----
                audioPath (str): Path to audio file
                begin (float): begin time in seconds
                end (float): end time in seconds
                fs (float): Sample rate
                normalize (bool): Normalize loaded audio to [-1, 1]
                target_fs (int): Target sample rate
            
            Returns
            -------
                result (np.ndarray): Audio buffer
                fs (float): Sample rate
        """
        # Load audio
        fs, data = self.audio_dict[audioPath]
        assert float(fs) == float(self.sample_rate),\
            f"Audio files do not have the expected sample rate: {fs} != {self.sample_rate}"
        b = math.floor(begin * fs)
        e = math.floor(end * fs)
        l = data.shape[0]

        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        result = (data[max(b,0): min(e,l), :])

        # handle padding
        lPad = max(-b, 0)
        rPad = max(e-l, 0)

        # normalize the audio to [-1,1] accoriding to the type
        # can move this normalize to init?? For speedup??
        if normalize:
            tMax = (np.iinfo(result.dtype)).max
            result = np.divide(result, tMax, dtype=np.float32)

        if lPad >0 or rPad>0:
            result = np.pad(result,  ((lPad, rPad),(0,0)), 'constant')
        return result, fs
   
# dset = TranskunDataset("../../../../testing_snap/transkundata/train", 44100, 8, 16)
# dset.fetchData(0, 0, 20, audioNormalize=True, notesStrictlyContained=True)
