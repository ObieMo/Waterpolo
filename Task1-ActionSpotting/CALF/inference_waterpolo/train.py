import os
import time

import numpy as np
import torch
from tqdm import tqdm

from json_io import predictions2json
from preprocessing import NMS, timestamps2long


def test(dataloader, model, device, save_predictions=False):
    spotting_predictions = []

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat, size) in t:
            _ = time.time() - end

            feat = feat.to(device).squeeze(0)
            feat = feat.unsqueeze(1)

            _, output_spotting = model(feat)
            timestamp_long = timestamps2long(
                output_spotting.cpu().detach(),
                int(size.item()),
                chunk_size,
                receptive_field,
            )
            spotting_predictions.append(timestamp_long)
            end = time.time()

    detections_numpy = []
    for detection in spotting_predictions:
        detections_numpy.append(NMS(detection.numpy(), 20 * model.framerate))

    if save_predictions:
        predictions2json(detections_numpy[0], os.path.join("inference_waterpolo", "outputs"), model.framerate)
