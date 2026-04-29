import argparse
import logging
import os
import pickle as pkl

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def list_match_dirs(dataset_path, split):
    split_root = os.path.join(dataset_path, split)
    if not os.path.isdir(split_root):
        logging.warning("Split directory does not exist, skipping: %s", split_root)
        return []

    return [
        os.path.join(split_root, name)
        for name in sorted(os.listdir(split_root))
        if os.path.isdir(os.path.join(split_root, name))
    ]


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_feature_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature file: {path}")

    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array in {path}, got shape {features.shape}")

    return features


def compute_average(feature_paths):
    feature_sum = None
    total_rows = 0
    feature_dim = None

    for path in tqdm(feature_paths, desc="Computing train feature mean"):
        features = load_feature_array(path)

        if feature_dim is None:
            feature_dim = features.shape[1]
        elif features.shape[1] != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimension in {path}: expected {feature_dim}, got {features.shape[1]}"
            )

        features64 = features.astype(np.float64, copy=False)
        current_sum = np.sum(features64, axis=0)

        if feature_sum is None:
            feature_sum = current_sum
        else:
            feature_sum += current_sum

        total_rows += features.shape[0]

    if total_rows == 0:
        raise ValueError("No training frames found to compute the PCA mean.")

    average = feature_sum / total_rows
    logging.info("Computed train mean from %s frames with %s dimensions.", total_rows, feature_dim)
    return average


def fit_pca(feature_paths, average, dim_reduction):
    centered_features = []

    for path in tqdm(feature_paths, desc="Loading train features for PCA"):
        features = load_feature_array(path)
        centered_features.append(features - average)

    pca_data = np.vstack(centered_features)

    if dim_reduction > pca_data.shape[1]:
        raise ValueError(
            f"dim_reduction={dim_reduction} exceeds feature dimension {pca_data.shape[1]}"
        )
    if dim_reduction > pca_data.shape[0]:
        raise ValueError(
            f"dim_reduction={dim_reduction} exceeds the number of training frames {pca_data.shape[0]}"
        )

    logging.info(
        "Fitting PCA on %s frames: %s -> %s dimensions.",
        pca_data.shape[0],
        pca_data.shape[1],
        dim_reduction,
    )
    pca = PCA(n_components=dim_reduction, svd_solver="full")
    pca.fit(pca_data)
    return pca


def save_pickle(obj, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite_pca to replace it explicitly."
        )

    ensure_parent_dir(path)
    with open(path, "wb") as fobj:
        pkl.dump(obj, fobj)


def transform_split(match_dirs, split_name, input_features, output_features, average, pca, overwrite=False):
    transformed = 0
    skipped = 0

    for match_dir in tqdm(match_dirs, desc=f"Transforming {split_name}"):
        input_path = os.path.join(match_dir, input_features)
        output_path = os.path.join(match_dir, output_features)

        if os.path.exists(output_path) and not overwrite:
            logging.info("Skipping existing reduced features: %s", output_path)
            skipped += 1
            continue

        features = load_feature_array(input_path)
        reduced = pca.transform(features - average)
        ensure_parent_dir(output_path)
        np.save(output_path, reduced)
        transformed += 1

    return transformed, skipped


def main(args):
    train_match_dirs = list_match_dirs(args.dataset_path, args.train_split)
    if len(train_match_dirs) == 0:
        raise ValueError(
            f"No match folders found in training split: {os.path.join(args.dataset_path, args.train_split)}"
        )

    train_feature_paths = [
        os.path.join(match_dir, args.input_features) for match_dir in train_match_dirs
    ]

    average = compute_average(train_feature_paths)
    pca = fit_pca(train_feature_paths, average, args.dim_reduction)

    save_pickle(average, args.scaler_file, overwrite=args.overwrite_pca)
    save_pickle(pca, args.pca_file, overwrite=args.overwrite_pca)
    logging.info("Saved train mean to %s", args.scaler_file)
    logging.info("Saved PCA model to %s", args.pca_file)

    for split in args.transform_splits:
        match_dirs = list_match_dirs(args.dataset_path, split)
        if len(match_dirs) == 0:
            continue

        logging.info("Applying PCA to split '%s' (%s matches).", split, len(match_dirs))
        transformed, skipped = transform_split(
            match_dirs,
            split_name=split,
            input_features=args.input_features,
            output_features=args.output_features,
            average=average,
            pca=pca,
            overwrite=args.overwrite_features,
        )
        logging.info(
            "Finished split '%s': transformed=%s skipped=%s",
            split,
            transformed,
            skipped,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit PCA on waterpolo train features and apply it to train/valid/test."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset root containing train/valid/test match folders.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Split used to fit the PCA statistics.",
    )
    parser.add_argument(
        "--transform_splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to transform with the fitted PCA.",
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default="features_raw.npy",
        help="Raw feature filename expected inside each match folder.",
    )
    parser.add_argument(
        "--output_features",
        type=str,
        default="features.npy",
        help="Reduced feature filename to write inside each match folder.",
    )
    parser.add_argument(
        "--pca_file",
        type=str,
        default="Features/waterpolo_pca_512.pkl",
        help="Output path for the fitted PCA pickle.",
    )
    parser.add_argument(
        "--scaler_file",
        type=str,
        default="Features/waterpolo_average_512.pkl",
        help="Output path for the fitted train-mean pickle.",
    )
    parser.add_argument(
        "--dim_reduction",
        type=int,
        default=512,
        help="Target PCA dimension.",
    )
    parser.add_argument(
        "--overwrite_pca",
        action="store_true",
        help="Overwrite existing PCA/scaler pickle files.",
    )
    parser.add_argument(
        "--overwrite_features",
        action="store_true",
        help="Overwrite existing reduced feature files.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Logging level.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), None),
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main(args)
