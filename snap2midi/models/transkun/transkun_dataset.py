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
    
    def load_audio(self, audioPath):
        fs, data = wavfile.read(audioPath, mmap = True)
        return fs, data
    
    def build_chunks(self, seed: float):
        print("Building chunks...")
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
            audioSlice = self.augmentator(audioSlice)

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
