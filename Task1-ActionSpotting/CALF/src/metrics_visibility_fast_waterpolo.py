import numpy as np

from metrics_visibility_fast import AverageMeter, NMS, delta_curve


def average_mAP(targets, detections, closests, framerate=2):
    mAP, mAP_per_class, _, _, _, _ = delta_curve(
        targets, closests, detections, framerate
    )

    integral = 0.0
    for i in np.arange(len(mAP) - 1):
        integral += 5 * (mAP[i] + mAP[i + 1]) / 2
    a_mAP = integral / (5 * (len(mAP) - 1))

    a_mAP_per_class = []
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class) - 1):
            integral_per_class += 5 * (mAP_per_class[i][c] + mAP_per_class[i + 1][c]) / 2
        a_mAP_per_class.append(integral_per_class / (5 * (len(mAP_per_class) - 1)))

    return a_mAP, a_mAP_per_class
