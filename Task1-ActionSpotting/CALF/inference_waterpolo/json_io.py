import json
import os

import numpy as np

from config.classes import INVERSE_EVENT_DICTIONARY_V2


def predictions2json(predictions, output_path, framerate=2):
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, "Predictions-v2.json")

    frames, classes = np.where(predictions >= 0)

    json_data = {"predictions": []}

    for frame_index, class_index in zip(frames, classes):
        confidence = predictions[frame_index, class_index]
        total_seconds = int(frame_index // framerate)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        prediction_data = {
            "gameTime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "label": INVERSE_EVENT_DICTIONARY_V2[class_index],
            "position": str(int((frame_index / framerate) * 1000)),
            "frameId": str(int(frame_index)),
            "confidence": str(confidence),
        }
        json_data["predictions"].append(prediction_data)

    with open(output_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4)
