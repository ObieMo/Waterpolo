import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

from config.classes_waterpolo import EVENT_DICTIONARY_V2
from model import ContextAwareModel


def build_waterpolo_model(args):
    return ContextAwareModel(
        weights=None,
        input_size=args.num_features,
        num_classes=len(EVENT_DICTIONARY_V2),
        chunk_size=args.chunk_size * args.framerate,
        dim_capsule=args.dim_capsule,
        receptive_field=args.receptive_field * args.framerate,
        num_detections=args.num_detections,
        framerate=args.framerate,
    )


def main(args):
    source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    source_state = source_checkpoint["state_dict"]

    model = build_waterpolo_model(args)
    target_state = model.state_dict()

    filtered_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }

    converted_checkpoint = dict(source_checkpoint)
    converted_checkpoint["state_dict"] = filtered_state
    converted_checkpoint["converted_from"] = args.source_checkpoint
    converted_checkpoint["num_classes"] = len(EVENT_DICTIONARY_V2)

    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save(converted_checkpoint, args.output_checkpoint)

    kept = sorted(filtered_state.keys())
    skipped = sorted(set(source_state.keys()) - set(filtered_state.keys()))

    print(f"Saved converted checkpoint to: {args.output_checkpoint}")
    print(f"Kept {len(kept)} tensors")
    print(f"Skipped {len(skipped)} tensors")
    if skipped:
        print("Skipped keys:")
        for key in skipped:
            print(f"  {key}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a SoccerNet CALF checkpoint into a water polo initialization checkpoint",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_checkpoint",
        required=True,
        type=str,
        help="Path to the original checkpoint",
    )
    parser.add_argument(
        "--output_checkpoint",
        required=False,
        type=str,
        default=os.path.join("models", "CALF_benchmark_waterpolo_init", "model.pth.tar"),
        help="Path to save the filtered checkpoint",
    )
    parser.add_argument("--num_features", required=False, type=int, default=512)
    parser.add_argument("--dim_capsule", required=False, type=int, default=16)
    parser.add_argument("--framerate", required=False, type=int, default=2)
    parser.add_argument("--chunk_size", required=False, type=int, default=120)
    parser.add_argument("--receptive_field", required=False, type=int, default=40)
    parser.add_argument("--num_detections", required=False, type=int, default=15)

    main(parser.parse_args())
