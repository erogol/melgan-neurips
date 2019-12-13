import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ids, mel_len, ap, eval=False):
        self.metadata = ids
        self.eval = eval
        self.mel_len = mel_len
        self.pad = 0 
        self.ap = ap

    def __getitem__(self, index):
        wav_path, mel_path = self.metadata[index]
        m = np.load(mel_path).astype('float32')
        x = self.ap.load_wav(wav_path)
        return m, x

    def __len__(self):
        return len(self.metadata)

    def collate(self, batch):
        min_mel_len = np.min([x[0].shape[-1] for x in batch])
        active_mel_len = np.minimum(min_mel_len, self.mel_len)
        seq_len = active_mel_len * self.ap.hop_length
        mel_win = active_mel_len
        max_offsets = [x[0].shape[-1] - mel_win for x in batch]
        if self.eval:
            mel_offsets = [10] * len(batch)
        else:
            mel_offsets = [np.random.randint(0, np.maximum(1, offset)) for offset in max_offsets]
        sig_offsets = [offset * self.ap.hop_length for offset in mel_offsets]

        mels = [
            x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win]
            for i, x in enumerate(batch)
        ]

        audio = [
            x[1][sig_offsets[i] : sig_offsets[i] + seq_len]
            for i, x in enumerate(batch)
        ]
        mels = np.stack(mels).astype(np.float32)
        mels = torch.FloatTensor(mels)
        audio = np.stack(audio).astype(np.float32)
        audio = torch.FloatTensor(audio)
        audio = audio[:, :seq_len]
        assert mels.shape[2] * self.ap.hop_length == audio.shape[1], f"{mels.shape[2] * self.ap.hop_length} vs {audio.shape[1]}"
        return audio, mels