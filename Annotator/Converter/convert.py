import json
from pathlib import Path

# Keep only these codes
GOAL_CODES = {"GA", "GE", "GC", "5"}
MISSED_SHOT_CODES = {"SA", "MX", "BR"}


def map_label(code: str):
    if code in GOAL_CODES:
        return "GOAL"
    if code in MISSED_SHOT_CODES:
        return "MissedShot"
    return None


def convert_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = []

    for row in data:
        code = row.get("code")
        label = map_label(code)

        # Ignore everything else
        if label is None:
            continue

        ann = {
            "gameTime": row["video_time_hhmmss"],
            "label": label,
            "position": str(row["frame_idx"]),
            "timeSec": row["sample_time_sec"],
            "code": code
        }
        annotations.append(ann)

    output = {"annotations": annotations}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def process_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.json"):
        out_file = output_dir / file.name
        convert_file(file, out_file)
        print(f"Converted {file} -> {out_file}")


if __name__ == "__main__":
    process_folder("input_jsons", "output_jsons")