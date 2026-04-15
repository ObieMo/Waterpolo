from torch.utils.data import Dataset

import json
import os
import random

import numpy as np
import torch

from config.classes_waterpolo import EVENT_DICTIONARY_V2, K_V2
from preprocessing import getChunks_anchors, getTimestampTargets, oneHotToShifts


def _list_match_dirs(path, split=None):
    """
    Return match directories.
    - If split is a list/tuple, it is interpreted as explicit match folder names.
    - If split is a string and <path>/<split> exists, matches are read from that subfolder.
    - Otherwise, matches are read directly from <path>.
    """
    if isinstance(split, (list, tuple)):
        return [os.path.join(path, name) for name in split]

    root = path
    if isinstance(split, str):
        split_root = os.path.join(path, split)
        if os.path.isdir(split_root):
            root = split_root

    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]


class WaterpoloClips(Dataset):
    """
    Clean CALF-compatible dataloader for one-timeline matches:
        dataset/
            match_x/
                features.npy
                Labels.json
    """

    def __init__(
        self,
        path,
        features="features.npy",
        labels="Labels.json",
        split=None,
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        event_dictionary=None,
        k_parameters=None,
        num_detections=5,
    ):
        self.path = path
        self.features = features
        self.labels = labels
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.framerate = framerate
        self.split = split

        self.dict_event = event_dictionary if event_dictionary is not None else EVENT_DICTIONARY_V2
        self.num_classes = len(self.dict_event)
        base_k = k_parameters if k_parameters is not None else K_V2
        self.K_parameters = base_k * framerate
        self.num_detections = num_detections

        self.match_dirs = _list_match_dirs(path, split=split)
        if len(self.match_dirs) == 0:
            raise ValueError(f"No match folders found in: {path}")
        self.match_names = [os.path.basename(match_dir) for match_dir in self.match_dirs]

        self.game_feats = []
        self.game_labels = []
        self.game_anchors = [[] for _ in np.arange(self.num_classes + 1)]


        params_np = self.K_parameters.detach().cpu().numpy()

        for game_index, match_dir in enumerate(self.match_dirs):
            features_path = os.path.join(match_dir, self.features)
            labels_path = os.path.join(match_dir, self.labels)

            feat = np.load(features_path)
            label_onehot = np.zeros((feat.shape[0], self.num_classes))

            if os.path.exists(labels_path):
                annotations = json.load(open(labels_path, "r")).get("annotations", [])
                for annotation in annotations:
                    event = annotation.get("label")
                    if event not in self.dict_event:
                        continue
                    class_idx = self.dict_event[event]

                    if "frameId" in annotation:
                        frame = int(annotation["frameId"])
                    elif "timeSec" in annotation:
                        frame = int(float(annotation["timeSec"]) * self.framerate)
                    else:
                        continue

                    frame = min(max(frame, 0), feat.shape[0] - 1)
                    label_onehot[frame, class_idx] = 1

            shifts = oneHotToShifts(label_onehot, params_np)
            anchors = getChunks_anchors(
                shifts,
                game_index,
                params_np,
                self.chunk_size,
                self.receptive_field,
            )

            self.game_feats.append(feat)
            self.game_labels.append(shifts)
            for anchor in anchors:
                self.game_anchors[anchor[2]].append(anchor)

        self.available_anchor_classes = [
            class_idx for class_idx, anchors in enumerate(self.game_anchors) if len(anchors) > 0
        ]
        if len(self.available_anchor_classes) == 0:
            raise ValueError("No anchors generated. Check labels, class dictionary, and chunk settings.")

    def __getitem__(self, index):
        # Keep training sampling strategy: pick a random anchor class then a random anchor.
        class_selection = random.choice(self.available_anchor_classes)
        event_selection = random.randint(0, len(self.game_anchors[class_selection]) - 1)
        game_index, anchor, _ = self.game_anchors[class_selection][event_selection]

        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size + self.receptive_field, -self.receptive_field)
            start = anchor + shift
        else:
            start = random.randint(anchor[0], anchor[1] - self.chunk_size)

        if start < 0:
            start = 0
        if start + self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0] - self.chunk_size - 1
        if start < 0:
            start = 0

        features_chunk = self.game_feats[game_index][start : start + self.chunk_size]
        segmentation_targets = self.game_labels[game_index][start : start + self.chunk_size].copy()

        half_rf = int(np.ceil(self.receptive_field / 2))
        segmentation_targets[:half_rf, :] = -1
        segmentation_targets[-half_rf:, :] = -1

        detection_targets = getTimestampTargets(
            np.array([segmentation_targets]), self.num_detections
        )[0]

        return (
            torch.from_numpy(features_chunk),
            torch.from_numpy(segmentation_targets),
            torch.from_numpy(detection_targets),
        )

    def __len__(self):
        return self.chunks_per_epoch


class WaterpoloClipsTesting(Dataset):
    """
    Testing/evaluation dataloader for one-timeline water polo matches.
    Returns all sliding clips for one match and the full-timeline labels.
    """

    def __init__(
        self,
        path,
        features="features.npy",
        labels="Labels.json",
        split=None,
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        event_dictionary=None,
        num_detections=5,
    ):
        self.path = path
        self.features = features
        self.labels = labels
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.split = split

        self.dict_event = event_dictionary if event_dictionary is not None else EVENT_DICTIONARY_V2
        self.num_classes = len(self.dict_event)
        self.K_parameters = K_V2 * framerate
        self.num_detections = num_detections

        self.match_dirs = _list_match_dirs(path, split=split)
        if len(self.match_dirs) == 0:
            raise ValueError(f"No match folders found in: {path}")
        self.match_names = [os.path.basename(match_dir) for match_dir in self.match_dirs]

    def __getitem__(self, index):
        match_dir = self.match_dirs[index]
        features_path = os.path.join(match_dir, self.features)
        labels_path = os.path.join(match_dir, self.labels)

        feat = np.load(features_path)
        label = np.zeros((feat.shape[0], self.num_classes))

        if os.path.exists(labels_path):
            annotations = json.load(open(labels_path, "r")).get("annotations", [])
            for annotation in annotations:
                event = annotation.get("label")
                if event not in self.dict_event:
                    continue
                class_idx = self.dict_event[event]

                if "frameId" in annotation:
                    frame = int(annotation["frameId"])
                elif "timeSec" in annotation:
                    frame = int(float(annotation["timeSec"]) * self.framerate)
                else:
                    continue

                frame = min(max(frame, 0), feat.shape[0] - 1)

                value = 1
                if "visibility" in annotation and annotation["visibility"] == "not shown":
                    value = -1

                label[frame, class_idx] = value

        def feats2clip(feats, stride, clip_length):
            idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx + i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0] - 1)
            idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length

            return feats[idx, :]

        feat_clips = feats2clip(
            torch.from_numpy(feat),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
        )

        return feat_clips, torch.from_numpy(label)

    def __len__(self):
        return len(self.match_dirs)
