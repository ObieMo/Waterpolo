import json
import os

import numpy as np


def predictions2json_single(
    predictions,
    output_path,
    game_info,
    framerate=2,
    inverse_event_dictionary=None,
):
    if inverse_event_dictionary is None:
        raise ValueError("inverse_event_dictionary is required for water polo export.")

    os.makedirs(os.path.join(output_path, game_info), exist_ok=True)
    output_file_path = os.path.join(output_path, game_info, "Predictions-v2.json")

    frames, class_indexes = np.where(predictions >= 0)

    json_data = {"UrlLocal": game_info, "predictions": []}

    for frame_index, class_index in zip(frames, class_indexes):
        confidence = predictions[frame_index, class_index]
        total_seconds = int(frame_index // framerate)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        prediction_data = {
            "gameTime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "label": inverse_event_dictionary[class_index],
            "position": str(int((frame_index / framerate) * 1000)),
            "frameId": str(int(frame_index)),
            "confidence": str(confidence),
        }
        json_data["predictions"].append(prediction_data)

    with open(output_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4)
