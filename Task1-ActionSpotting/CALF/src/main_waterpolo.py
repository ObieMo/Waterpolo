import logging
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import torch

from dataset_waterpolo import WaterpoloClips, WaterpoloClipsTesting
from loss import ContextAwareLoss, SpottingLoss
from model import ContextAwareModel
from train_waterpolo import test, trainer

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def resolve_load_weights_path(model_name, load_weights):
    if load_weights in (None, ""):
        return load_weights
    return os.path.join("models", model_name, "checkpoints", f"{load_weights}.pth.tar")


def main(args):
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            logging.warning(
                "TensorBoard logging requested, but torch.utils.tensorboard is unavailable in this environment."
            )
        else:
            log_dir = os.path.join("models", args.model_name, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            logging.info("TensorBoard logs will be written to %s", log_dir)

    try:
        if not args.test_only:
            dataset_Train = WaterpoloClips(
                path=args.dataset_path,
                features=args.features,
                labels=args.labels_file,
                split="train",
                framerate=args.framerate,
                chunk_size=args.chunk_size * args.framerate,
                receptive_field=args.receptive_field * args.framerate,
                chunks_per_epoch=args.chunks_per_epoch,
            )
            dataset_Valid = WaterpoloClips(
                path=args.dataset_path,
                features=args.features,
                labels=args.labels_file,
                split="valid",
                framerate=args.framerate,
                chunk_size=args.chunk_size * args.framerate,
                receptive_field=args.receptive_field * args.framerate,
                chunks_per_epoch=args.chunks_per_epoch,
            )
            dataset_Valid_metric = WaterpoloClipsTesting(
                path=args.dataset_path,
                features=args.features,
                labels=args.labels_file,
                split="valid",
                framerate=args.framerate,
                chunk_size=args.chunk_size * args.framerate,
                receptive_field=args.receptive_field * args.framerate,
            )

        dataset_Test = WaterpoloClipsTesting(
            path=args.dataset_path,
            features=args.features,
            labels=args.labels_file,
            split="test",
            framerate=args.framerate,
            chunk_size=args.chunk_size * args.framerate,
            receptive_field=args.receptive_field * args.framerate,
        )

        model = ContextAwareModel(
            weights=args.load_weights,
            input_size=args.num_features,
            num_classes=dataset_Test.num_classes,
            chunk_size=args.chunk_size * args.framerate,
            dim_capsule=args.dim_capsule,
            receptive_field=args.receptive_field * args.framerate,
            num_detections=dataset_Test.num_detections,
            framerate=args.framerate,
        ).cuda()
        logging.info(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Total number of parameters: " + str(total_params))

        if not args.test_only:
            train_loader = torch.utils.data.DataLoader(
                dataset_Train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.max_num_worker,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset_Valid,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.max_num_worker,
                pin_memory=True,
            )
            val_metric_loader = torch.utils.data.DataLoader(
                dataset_Valid_metric, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
            )

        test_loader = torch.utils.data.DataLoader(
            dataset_Test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

        if not args.test_only:
            criterion_segmentation = ContextAwareLoss(K=dataset_Train.K_parameters)
            criterion_spotting = SpottingLoss(
                lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.LR,
                betas=(0.9, 0.999),
                eps=1e-07,
                weight_decay=0,
                amsgrad=False,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", verbose=True, patience=args.patience
            )

            start_epoch = 0
            if args.load_weights is not None:
                checkpoint = torch.load(args.load_weights)
                restore_optimizer = "optimizer" in checkpoint and "converted_from" not in checkpoint
                if restore_optimizer:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    except (RuntimeError, ValueError) as exc:
                        logging.warning(
                            "Skipping optimizer state from %s because it is incompatible with the current model: %s",
                            args.load_weights,
                            exc,
                        )
                elif "optimizer" in checkpoint:
                    logging.info(
                        "Skipping optimizer state from %s because this checkpoint is a converted initialization checkpoint.",
                        args.load_weights,
                    )
                start_epoch = checkpoint.get("epoch", 0)
                logging.info("Resuming from epoch %s", start_epoch)

            trainer(
                train_loader,
                val_loader,
                val_metric_loader,
                test_loader,
                model,
                optimizer,
                scheduler,
                [criterion_segmentation, criterion_spotting],
                [args.loss_weight_segmentation, args.loss_weight_detection],
                model_name=args.model_name,
                writer=writer,
                max_epochs=args.max_epochs,
                evaluation_frequency=args.evaluation_frequency,
                start_epoch=start_epoch,
            )

        checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        a_mAP, a_mAP_per_class = test(
            test_loader, model=model, model_name=args.model_name, save_predictions=True
        )
        logging.info("Best performance at end of training ")
        logging.info("Average mAP: " + str(a_mAP))
        logging.info("Average mAP per class: " + str(a_mAP_per_class))
        if writer is not None:
            writer.add_scalar("mAP/test_final", a_mAP, 0)

        return a_mAP
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="context aware loss function - waterpolo",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Path to dataset root with train/valid/test"
    )
    parser.add_argument(
        "--features", required=False, type=str, default="features.npy", help="Feature filename"
    )
    parser.add_argument(
        "--labels_file", required=False, type=str, default="Labels.json", help="Label filename"
    )
    parser.add_argument("--max_epochs", required=False, type=int, default=20)
    parser.add_argument("--load_weights", required=False, type=str, default=None)
    parser.add_argument("--model_name", required=False, type=str, default="CALF_Waterpolo")
    parser.add_argument("--test_only", required=False, action="store_true")

    parser.add_argument("--num_features", required=False, type=int, default=512)
    parser.add_argument("--chunks_per_epoch", required=False, type=int, default=18000)
    parser.add_argument("--evaluation_frequency", required=False, type=int, default=20)
    parser.add_argument("--dim_capsule", required=False, type=int, default=16)
    parser.add_argument("--framerate", required=False, type=int, default=2)
    parser.add_argument("--chunk_size", required=False, type=int, default=120)
    parser.add_argument("--receptive_field", required=False, type=int, default=40)
    parser.add_argument("--lambda_coord", required=False, type=float, default=5.0)
    parser.add_argument("--lambda_noobj", required=False, type=float, default=0.5)
    parser.add_argument("--loss_weight_segmentation", required=False, type=float, default=0.000367)
    parser.add_argument("--loss_weight_detection", required=False, type=float, default=1.0)

    parser.add_argument("--batch_size", required=False, type=int, default=32)
    parser.add_argument("--LR", required=False, type=float, default=1e-03)
    parser.add_argument("--patience", required=False, type=int, default=25)
    parser.add_argument("--GPU", required=False, type=int, default=-1)
    parser.add_argument("--max_num_worker", required=False, type=int, default=4)
    parser.add_argument("--tensorboard", required=False, action="store_true")
    parser.add_argument("--loglevel", required=False, type=str, default="INFO")

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
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

    args.load_weights = resolve_load_weights_path(args.model_name, args.load_weights)

    start = time.time()
    logging.info("Starting main function")
    main(args)
    logging.info(f"Total Execution Time is {time.time()-start} seconds")
