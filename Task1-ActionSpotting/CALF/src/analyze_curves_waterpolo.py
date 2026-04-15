import csv
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from config.classes_waterpolo import INVERSE_EVENT_DICTIONARY_V2
from dataset_waterpolo import WaterpoloClipsTesting
from metrics_visibility_fast import NMS, compute_class_scores, compute_mAP, compute_precision_recall_curve
from model import ContextAwareModel
from preprocessing import timestamps2long

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def resolve_load_weights_path(model_name, load_weights):
    if load_weights in (None, ""):
        return None
    if isinstance(load_weights, str) and load_weights.strip().lower() in {"none", "null"}:
        return None
    return os.path.join("models", model_name, "checkpoints", f"{load_weights}.pth.tar")


def sanitize_delta_seconds_list(delta_seconds_list):
    values = sorted(set(float(value) for value in delta_seconds_list))
    if not values:
        raise ValueError("delta_seconds_list must contain at least one value.")
    return values


def load_model(args, dataset, device):
    model = ContextAwareModel(
        weights=args.load_weights,
        input_size=args.num_features,
        num_classes=dataset.num_classes,
        chunk_size=args.chunk_size * args.framerate,
        dim_capsule=args.dim_capsule,
        receptive_field=args.receptive_field * args.framerate,
        num_detections=dataset.num_detections,
        framerate=args.framerate,
    ).to(device)

    if args.load_weights is None:
        best_checkpoint = os.path.join("models", args.model_name, "model.pth.tar")
        if os.path.exists(best_checkpoint):
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logging.info("Loaded best checkpoint from %s", best_checkpoint)
        else:
            logging.info("No checkpoint requested or found; analyzing random initialization.")

    return model


def collect_targets_closests_detections(dataloader, model, device):
    targets_numpy = []
    closests_numpy = []
    detections_numpy = []

    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as progress:
            for _, (feat_clips, label) in progress:
                feat_clips = feat_clips.to(device).squeeze(0)
                label = label.float().squeeze(0)
                feat_clips = feat_clips.unsqueeze(1)

                _, output_spotting = model(feat_clips)
                timestamp_long = timestamps2long(
                    output_spotting.cpu().detach(),
                    label.size()[0],
                    model.chunk_size,
                    model.receptive_field,
                )

                target_numpy = label.numpy()
                detection_numpy = NMS(timestamp_long.numpy(), 40 * model.framerate)
                closest_numpy = np.zeros(target_numpy.shape) - 1

                for class_index in np.arange(target_numpy.shape[-1]):
                    indexes = np.where(target_numpy[:, class_index] != 0)[0].tolist()
                    if len(indexes) == 0:
                        continue
                    indexes.insert(0, -indexes[0])
                    indexes.append(2 * closest_numpy.shape[0])
                    for idx in np.arange(len(indexes) - 2) + 1:
                        start = max(0, (indexes[idx - 1] + indexes[idx]) // 2)
                        stop = min(closest_numpy.shape[0], (indexes[idx] + indexes[idx + 1]) // 2)
                        closest_numpy[start:stop, class_index] = target_numpy[indexes[idx], class_index]

                targets_numpy.append(target_numpy)
                closests_numpy.append(closest_numpy)
                detections_numpy.append(detection_numpy)

    return targets_numpy, closests_numpy, detections_numpy


def get_actual_thresholds(detections):
    scores = []
    for detection in detections:
        valid_scores = detection[detection >= 0]
        if valid_scores.size:
            scores.extend(float(score) for score in valid_scores.tolist())

    if not scores:
        return np.array([0.0], dtype=float)

    unique_scores = np.array(sorted(set(scores)), dtype=float)
    epsilon = 1e-6
    return np.concatenate([unique_scores, [unique_scores[-1] + epsilon]])


def build_class_detection_tables(targets, closests, detections, delta_frames):
    num_classes = targets[0].shape[-1]
    class_tables = []

    for class_index in np.arange(num_classes):
        total_detections = np.zeros((1, 3))
        total_detections[0, 0] = -1
        n_gt_labels = 0

        for target, closest, detection in zip(targets, closests, detections):
            tmp_detections, tmp_n_gt_visible, tmp_n_gt_unshown = compute_class_scores(
                target[:, class_index],
                closest[:, class_index],
                detection[:, class_index],
                delta_frames,
            )
            total_detections = np.append(total_detections, tmp_detections, axis=0)
            n_gt_labels += tmp_n_gt_visible + tmp_n_gt_unshown

        class_tables.append(
            {
                "detections": total_detections,
                "gt_count": n_gt_labels,
            }
        )

    return class_tables


def compute_threshold_rows(class_tables, thresholds):
    rows = []

    for threshold in thresholds:
        class_precisions = []
        class_recalls = []
        tp_total = 0.0
        pred_total = 0
        gt_total = 0

        for table in class_tables:
            detections = table["detections"]
            gt_count = table["gt_count"]
            pred_indexes = np.where(detections[:, 0] >= threshold)[0]
            tp = float(np.sum(detections[pred_indexes, 1]))
            pred_count = int(len(pred_indexes))

            precision = 0.0 if pred_count == 0 else float(tp / pred_count)
            recall = 0.0 if gt_count == 0 else float(tp / gt_count)

            class_precisions.append(precision)
            class_recalls.append(recall)
            tp_total += tp
            pred_total += pred_count
            gt_total += gt_count

        macro_precision = float(np.mean(class_precisions))
        macro_recall = float(np.mean(class_recalls))
        if macro_precision + macro_recall == 0:
            macro_f1 = 0.0
        else:
            macro_f1 = float(2 * macro_precision * macro_recall / (macro_precision + macro_recall))

        micro_precision = 0.0 if pred_total == 0 else float(tp_total / pred_total)
        micro_recall = 0.0 if gt_total == 0 else float(tp_total / gt_total)
        if micro_precision + micro_recall == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))

        rows.append(
            {
                "threshold": float(threshold),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "predictions_kept": pred_total,
                "tp_total": tp_total,
                "gt_total": gt_total,
            }
        )

    best_index = int(np.argmax([row["macro_f1"] for row in rows]))
    return rows, best_index


def compute_delta_map(targets, closests, detections, delta_frames):
    precision, recall, _, _, _, _ = compute_precision_recall_curve(
        targets, closests, detections, delta_frames
    )
    mAP_value, per_class = compute_mAP(precision, recall)
    return float(mAP_value), [float(value) for value in per_class]


def compute_delta_analysis(targets, closests, detections, framerate, delta_seconds_list):
    thresholds = get_actual_thresholds(detections)
    results = []

    for delta_seconds in delta_seconds_list:
        delta_frames = int(delta_seconds * framerate)
        class_tables = build_class_detection_tables(targets, closests, detections, delta_frames)
        threshold_rows, best_index = compute_threshold_rows(class_tables, thresholds)
        mAP_value, mAP_per_class = compute_delta_map(targets, closests, detections, delta_frames)
        results.append(
            {
                "delta_seconds": float(delta_seconds),
                "delta_frames": int(delta_frames),
                "tolerance_seconds_pm": float(delta_seconds / 2.0),
                "threshold_rows": threshold_rows,
                "best_index": best_index,
                "mAP": mAP_value,
                "mAP_per_class": mAP_per_class,
            }
        )

    return results


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_class_map_fields(mAP_per_class):
    fields = {}
    for class_index, value in enumerate(mAP_per_class):
        class_name = INVERSE_EVENT_DICTIONARY_V2.get(class_index, f"class_{class_index}")
        fields[f"mAP_{class_name}"] = value
    return fields


def maybe_plot_threshold_curves(output_dir, threshold_rows, best_index, delta_seconds):
    if plt is None:
        return

    thresholds = [row["threshold"] for row in threshold_rows]
    macro_precision = [row["macro_precision"] for row in threshold_rows]
    macro_recall = [row["macro_recall"] for row in threshold_rows]
    macro_f1 = [row["macro_f1"] for row in threshold_rows]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(thresholds, macro_precision, label="Macro Precision")
    plt.plot(thresholds, macro_recall, label="Macro Recall")
    plt.plot(thresholds, macro_f1, label="Macro F1")
    plt.axvline(
        thresholds[best_index],
        linestyle="--",
        color="black",
        label="Best Macro F1 Threshold",
    )
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Sweep (delta={int(delta_seconds)}s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_curves.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(macro_recall, macro_precision)
    plt.xlabel("Macro Recall")
    plt.ylabel("Macro Precision")
    plt.title(f"Precision-Recall Curve (delta={int(delta_seconds)}s)")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=200)
    plt.close(fig)


def maybe_plot_map_vs_delta(output_dir, delta_results):
    if plt is None:
        return

    tolerance_seconds = [row["tolerance_seconds_pm"] for row in delta_results]
    mAP_values = [row["mAP"] for row in delta_results]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(tolerance_seconds, mAP_values, marker="o")
    plt.xlabel("Tolerance (+/- seconds)")
    plt.ylabel("mAP")
    plt.title("mAP vs Temporal Tolerance")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "map_vs_tolerance.png"), dpi=200)
    plt.close(fig)


def save_outputs(output_dir, delta_results):
    map_rows = []
    best_threshold_rows = []

    for result in delta_results:
        delta_label = int(result["delta_seconds"])
        delta_dir = os.path.join(output_dir, f"delta_{delta_label}s")
        os.makedirs(delta_dir, exist_ok=True)

        write_csv(
            os.path.join(delta_dir, "threshold_metrics.csv"),
            [
                "threshold",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "predictions_kept",
                "tp_total",
                "gt_total",
            ],
            result["threshold_rows"],
        )

        best_row = result["threshold_rows"][result["best_index"]]
        with open(os.path.join(delta_dir, "summary.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"delta_seconds={result['delta_seconds']}\n")
            handle.write(f"delta_frames={result['delta_frames']}\n")
            handle.write(f"tolerance_seconds_pm={result['tolerance_seconds_pm']}\n")
            handle.write(f"mAP_at_delta={result['mAP']:.6f}\n")
            handle.write(f"mAP_per_class={result['mAP_per_class']}\n")
            for class_index, value in enumerate(result["mAP_per_class"]):
                class_name = INVERSE_EVENT_DICTIONARY_V2.get(class_index, f"class_{class_index}")
                handle.write(f"mAP_{class_name}={value:.6f}\n")
            handle.write(f"best_threshold_by_macro_f1={best_row['threshold']:.6f}\n")
            handle.write(f"best_macro_precision={best_row['macro_precision']:.6f}\n")
            handle.write(f"best_macro_recall={best_row['macro_recall']:.6f}\n")
            handle.write(f"best_macro_f1={best_row['macro_f1']:.6f}\n")
            handle.write(f"best_micro_precision={best_row['micro_precision']:.6f}\n")
            handle.write(f"best_micro_recall={best_row['micro_recall']:.6f}\n")
            handle.write(f"best_micro_f1={best_row['micro_f1']:.6f}\n")
            handle.write(f"predictions_kept_at_best_threshold={best_row['predictions_kept']}\n")
            handle.write(f"tp_total_at_best_threshold={best_row['tp_total']}\n")
            handle.write(f"gt_total={best_row['gt_total']}\n")

        maybe_plot_threshold_curves(
            delta_dir,
            result["threshold_rows"],
            result["best_index"],
            result["delta_seconds"],
        )

        map_rows.append(
            {
                "delta_seconds": result["delta_seconds"],
                "delta_frames": result["delta_frames"],
                "tolerance_seconds_pm": result["tolerance_seconds_pm"],
                "mAP": result["mAP"],
                "mAP_per_class": result["mAP_per_class"],
                **get_class_map_fields(result["mAP_per_class"]),
            }
        )
        best_threshold_rows.append(
            {
                "delta_seconds": result["delta_seconds"],
                "delta_frames": result["delta_frames"],
                "tolerance_seconds_pm": result["tolerance_seconds_pm"],
                "mAP": result["mAP"],
                **get_class_map_fields(result["mAP_per_class"]),
                "best_threshold_by_macro_f1": best_row["threshold"],
                "best_macro_precision": best_row["macro_precision"],
                "best_macro_recall": best_row["macro_recall"],
                "best_macro_f1": best_row["macro_f1"],
                "best_micro_precision": best_row["micro_precision"],
                "best_micro_recall": best_row["micro_recall"],
                "best_micro_f1": best_row["micro_f1"],
                "predictions_kept_at_best_threshold": best_row["predictions_kept"],
                "tp_total_at_best_threshold": best_row["tp_total"],
                "gt_total": best_row["gt_total"],
            }
        )

    write_csv(
        os.path.join(output_dir, "map_vs_delta.csv"),
        [
            "delta_seconds",
            "delta_frames",
            "tolerance_seconds_pm",
            "mAP",
            "mAP_per_class",
            *get_class_map_fields(delta_results[0]["mAP_per_class"]).keys(),
        ],
        map_rows,
    )
    write_csv(
        os.path.join(output_dir, "best_thresholds_by_delta.csv"),
        [
            "delta_seconds",
            "delta_frames",
            "tolerance_seconds_pm",
            "mAP",
            *get_class_map_fields(delta_results[0]["mAP_per_class"]).keys(),
            "best_threshold_by_macro_f1",
            "best_macro_precision",
            "best_macro_recall",
            "best_macro_f1",
            "best_micro_precision",
            "best_micro_recall",
            "best_micro_f1",
            "predictions_kept_at_best_threshold",
            "tp_total_at_best_threshold",
            "gt_total",
        ],
        best_threshold_rows,
    )

    return map_rows, best_threshold_rows


def main(args):
    args.delta_seconds_list = sanitize_delta_seconds_list(args.delta_seconds_list)

    device = torch.device("cuda" if args.GPU >= 0 and torch.cuda.is_available() else "cpu")
    dataset = WaterpoloClipsTesting(
        path=args.dataset_path,
        features=args.features,
        labels=args.labels_file,
        split=args.split,
        framerate=args.framerate,
        chunk_size=args.chunk_size * args.framerate,
        receptive_field=args.receptive_field * args.framerate,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = load_model(args, dataset, device)
    targets, closests, detections = collect_targets_closests_detections(dataloader, model, device)
    delta_results = compute_delta_analysis(
        targets,
        closests,
        detections,
        args.framerate,
        args.delta_seconds_list,
    )

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("models", args.model_name, "analysis", f"{args.split}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    map_rows, best_threshold_rows = save_outputs(output_dir, delta_results)
    maybe_plot_map_vs_delta(output_dir, delta_results)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"split={args.split}\n")
        handle.write(
            "delta_seconds_list=" + ",".join(str(int(value)) for value in args.delta_seconds_list) + "\n"
        )
        for row in best_threshold_rows:
            line = (
                f"delta={int(row['delta_seconds'])}s"
                f", mAP={row['mAP']:.6f}"
                + "".join(
                    f", {field}={row[field]:.6f}"
                    for field in get_class_map_fields(delta_results[0]["mAP_per_class"]).keys()
                )
                + f", best_threshold_by_macro_f1={row['best_threshold_by_macro_f1']:.6f}"
                + f", best_macro_f1={row['best_macro_f1']:.6f}"
                + "\n"
            )
            handle.write(line)

    logging.info("Saved analysis files to %s", output_dir)
    for row in best_threshold_rows:
        logging.info(
            "delta=%ss | mAP=%.4f | best_threshold=%.6f | best_macro_f1=%.4f",
            int(row["delta_seconds"]),
            row["mAP"],
            row["best_threshold_by_macro_f1"],
            row["best_macro_f1"],
        )
    if plt is None:
        logging.info(
            "matplotlib is not installed in this environment; CSVs and summaries were saved but PNG plots were skipped."
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Analyze confidence-threshold and delta curves for waterpolo spotting.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--split", required=False, type=str, default="valid")
    parser.add_argument("--features", required=False, type=str, default="features.npy")
    parser.add_argument("--labels_file", required=False, type=str, default="Labels.json")
    parser.add_argument("--model_name", required=False, type=str, default="CALF_Waterpolo")
    parser.add_argument("--load_weights", required=False, type=str, default=None)
    parser.add_argument("--num_features", required=False, type=int, default=512)
    parser.add_argument("--dim_capsule", required=False, type=int, default=16)
    parser.add_argument("--framerate", required=False, type=int, default=2)
    parser.add_argument("--chunk_size", required=False, type=int, default=120)
    parser.add_argument("--receptive_field", required=False, type=int, default=40)
    parser.add_argument(
        "--delta_seconds_list",
        nargs="+",
        required=False,
        type=float,
        default=[5, 10, 15, 20, 25, 30],
        help="Deltas in seconds to analyze. Real matching tolerance is +/- delta/2.",
    )
    parser.add_argument("--GPU", required=False, type=int, default=-1)
    parser.add_argument("--loglevel", required=False, type=str, default="INFO")

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)-5.5s] %(message)s")

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    args.load_weights = resolve_load_weights_path(args.model_name, args.load_weights)
    main(args)
