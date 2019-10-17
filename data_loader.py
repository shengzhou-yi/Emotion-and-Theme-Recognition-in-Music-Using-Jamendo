import os
import random
import numpy as np
import pickle
from torch.utils import data

def random_crop(size):
    def f(sound):
        org_size = sound.shape[1]
        start = random.randint(0, org_size - size)
        return sound[:, start: start + size]
    return f

def multi_crop(size, n_crops):
    def f(sound):
        sounds = np.zeros((n_crops, 96, size))
        org_size = sound.shape[1]
        stride = (org_size - size) // (n_crops - 1)
        for i in range(n_crops):
            sounds[i] = sound[:, stride * i : stride * i + size]
        return sounds
    return f

class AudioFolder(data.Dataset):
    def __init__(self, bc_learning, root, subset, tr_val='train', split=0, segment_length=1366):
        self.bc_learning = (bc_learning and tr_val == 'train')
        self.root = root
        self.tr_val = tr_val
        fn = '../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)

        val_seg = 10
        test_seg = val_seg * 2
        if tr_val == 'train':
            self.func = random_crop(segment_length)
        elif tr_val == 'validation':
            self.func = multi_crop(segment_length, val_seg)
            #self.func = random_crop(segment_length)
        elif tr_val == 'test':
            self.func = multi_crop(segment_length, test_seg)

    def __getitem__(self, index):
        if self.bc_learning is False:
            fn = os.path.join(self.root, 'download_melspecs', self.dictionary[index]['path'][:-3]+'npy')
            audio = self.func(np.load(fn))
            tags = self.dictionary[index]['tags']
        elif self.bc_learning is True:
            index1 = random.randint(0, len(self.dictionary) - 1)
            index2 = random.randint(0, len(self.dictionary) - 1)
            fn1 = os.path.join(self.root, 'download_melspecs', self.dictionary[index1]['path'][:-3]+'npy')
            fn2 = os.path.join(self.root, 'download_melspecs', self.dictionary[index2]['path'][:-3]+'npy')
            audio1 = self.func(np.load(fn1))
            audio2 = self.func(np.load(fn2))
            tags1 = self.dictionary[index1]['tags']
            tags2 = self.dictionary[index2]['tags']
            r = random.random()
            audio = r * audio1 + (1 - r) * audio2
            tags = r * tags1 + (1 - r) * tags2

        return audio.astype('float32'), tags.astype('float32'), self.dictionary[index]['path']

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)

def get_audio_loader(bc_learning, root, subset, batch_size, segment_length, shuffle, tr_val='train', split=0, num_workers=0):
    data_loader = data.DataLoader(dataset=AudioFolder(bc_learning, root, subset, tr_val, split, segment_length),
                                  batch_size = batch_size,
                                  shuffle = shuffle,
                                  num_workers=num_workers)
    return data_loader