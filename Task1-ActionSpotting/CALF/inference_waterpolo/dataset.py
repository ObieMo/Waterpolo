from torch.utils.data import Dataset

import numpy as np
import torch


class WaterpoloClipsTesting(Dataset):
    def __init__(self, path, framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.num_classes = 2
        self.num_detections = 15

    def __getitem__(self, index):
        feat = np.load(self.path)
        size = feat.shape[0]

        def feats2clip(feats, stride, clip_length):
            idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx + i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0] - 1)
            idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length

            return feats[idx, :]

        feat = feats2clip(
            torch.from_numpy(feat),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
        )

        return feat, size

    def __len__(self):
        return 1
