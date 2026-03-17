import numpy as np
import torch


def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):
    start = 0
    last = False
    receptive_field = receptive_field // 2

    timestamps_long = torch.zeros(
        [video_size, output_spotting.size()[-1] - 2],
        dtype=torch.float,
        device=output_spotting.device,
    ) - 1

    for batch in np.arange(output_spotting.size()[0]):
        tmp_timestamps = torch.zeros(
            [chunk_size, output_spotting.size()[-1] - 2],
            dtype=torch.float,
            device=output_spotting.device,
        ) - 1

        for i in np.arange(output_spotting.size()[1]):
            tmp_timestamps[
                torch.floor(output_spotting[batch, i, 1] * (chunk_size - 1)).type(torch.int),
                torch.argmax(output_spotting[batch, i, 2:]).type(torch.int),
            ] = output_spotting[batch, i, 0]

        if start == 0:
            timestamps_long[0 : chunk_size - receptive_field] = tmp_timestamps[0 : chunk_size - receptive_field]
        elif last:
            timestamps_long[start + receptive_field : start + chunk_size] = tmp_timestamps[receptive_field:]
            break
        else:
            timestamps_long[start + receptive_field : start + chunk_size - receptive_field] = tmp_timestamps[
                receptive_field : chunk_size - receptive_field
            ]

        start += chunk_size - 2 * receptive_field
        if start + chunk_size >= video_size:
            start = video_size - chunk_size
            last = True
    return timestamps_long


def NMS(detections, delta):
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape) - 1

    for i in np.arange(detections.shape[-1]):
        while np.max(detections_tmp[:, i]) >= 0:
            max_value = np.max(detections_tmp[:, i])
            max_index = np.argmax(detections_tmp[:, i])

            detections_NMS[max_index, i] = max_value
            detections_tmp[
                int(np.maximum(-(delta / 2) + max_index, 0)) : int(
                    np.minimum(max_index + int(delta / 2), detections.shape[0])
                ),
                i,
            ] = -1

    return detections_NMS
