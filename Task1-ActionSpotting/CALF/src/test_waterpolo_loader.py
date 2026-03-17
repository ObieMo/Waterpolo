from dataset_waterpolo import WaterpoloClips


def main():
    dataset = WaterpoloClips(
        path=r"C:\Users\Obie\Desktop\testdata",
        features="features.npy",
        labels="Labels.json",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=100,
    )

    print("len:", len(dataset))
    x, y_seg, y_det = dataset[0]
    print("x:", tuple(x.shape))
    print("y_seg:", tuple(y_seg.shape))
    print("y_det:", tuple(y_det.shape))


if __name__ == "__main__":
    main()
