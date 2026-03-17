import logging
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import torch

from dataset import WaterpoloClipsTesting
from train import test

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ContextAwareModel  # noqa: E402

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def main(args):
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    dataset_test = WaterpoloClipsTesting(
        path=args.features_path,
        framerate=args.framerate,
        chunk_size=args.chunk_size * args.framerate,
        receptive_field=args.receptive_field * args.framerate,
    )
    use_cuda = args.GPU >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = ContextAwareModel(
        weights=args.load_weights,
        input_size=args.num_features,
        num_classes=dataset_test.num_classes,
        chunk_size=args.chunk_size * args.framerate,
        dim_capsule=args.dim_capsule,
        receptive_field=args.receptive_field * args.framerate,
        num_detections=dataset_test.num_detections,
        framerate=args.framerate,
    ).to(device)
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_params))

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda
    )

    checkpoint = torch.load(
        os.path.join("models", args.model_name, "model.pth.tar"),
        map_location=device,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    test(test_loader, model=model, device=device, save_predictions=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="waterpolo inference",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--features_path", required=True, type=str, help="Path to features.npy")
    parser.add_argument("--features", required=False, type=str, default="features.npy", help="Unused, kept for CLI parity")
    parser.add_argument("--load_weights", required=False, type=str, default=None, help="weights to load")
    parser.add_argument("--model_name", required=False, type=str, default="CALF_Waterpolo", help="named of the model to save")
    parser.add_argument("--num_features", required=False, type=int, default=512, help="Number of input features")
    parser.add_argument("--dim_capsule", required=False, type=int, default=16, help="Dimension of the capsule network")
    parser.add_argument("--framerate", required=False, type=int, default=2, help="Framerate of the input features")
    parser.add_argument("--chunk_size", required=False, type=int, default=120, help="Size of the chunk (in seconds)")
    parser.add_argument("--receptive_field", required=False, type=int, default=40, help="Temporal receptive field of the network (in seconds)")
    parser.add_argument("--GPU", required=False, type=int, default=-1, help="ID of the GPU to use")
    parser.add_argument("--loglevel", required=False, type=str, default="INFO", help="logging level")

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    os.makedirs(os.path.join("inference_waterpolo", "outputs"), exist_ok=True)
    log_path = os.path.join(
        "models", args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info("Starting main function")
    main(args)
    logging.info(f"Total Execution Time is {time.time()-start} seconds")
