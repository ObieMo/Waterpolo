import logging
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from json_io_waterpolo import predictions2json_single
from metrics_visibility_fast_waterpolo import AverageMeter, NMS, average_mAP
from preprocessing import batch2long, timestamps2long


def trainer(
    train_loader,
    val_loader,
    val_metric_loader,
    test_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    weights,
    model_name,
    writer=None,
    max_epochs=1,
    evaluation_frequency=20,
    start_epoch=0,
):
    logging.info("start training")

    best_loss = 9e99
    best_metric = -1

    checkpoint_dir = os.path.join("models", model_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        loss_training = train(
            train_loader, model, criterion, weights, optimizer, epoch + 1, train=True
        )
        loss_validation = train(
            val_loader, model, criterion, weights, optimizer, epoch + 1, train=False
        )

        if writer is not None:
            writer.add_scalar("loss/train", loss_training, epoch + 1)
            writer.add_scalar("loss/val", loss_validation, epoch + 1)
            writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)
        torch.save(state, os.path.join(checkpoint_dir, "latest.pth.tar"))

        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        if is_better and evaluation_frequency > max_epochs:
            torch.save(state, best_model_path)

        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(val_metric_loader, model, model_name)[0]
            logging.info(
                "Validation performance at epoch %s -> %s",
                str(epoch + 1),
                str(performance_validation),
            )
            if writer is not None:
                writer.add_scalar("mAP/val", performance_validation, epoch + 1)

            is_better_metric = performance_validation > best_metric
            best_metric = max(performance_validation, best_metric)

            if is_better_metric and evaluation_frequency <= max_epochs:
                torch.save(state, best_model_path)
                performance_test = test(
                    test_loader, model, model_name, save_predictions=True
                )[0]
                logging.info(
                    "Test performance at epoch %s -> %s",
                    str(epoch + 1),
                    str(performance_test),
                )
                if writer is not None:
                    writer.add_scalar("mAP/test", performance_test, epoch + 1)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(loss_validation)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr is not prev_lr and scheduler.num_bad_epochs == 0:
            logging.info("Plateau Reached!")

        if prev_lr < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            break


def train(dataloader, model, criterion, weights, optimizer, epoch, train=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats, labels, targets) in t:
            data_time.update(time.time() - end)

            feats = feats.cuda()
            labels = labels.cuda().float()
            targets = targets.cuda().float()

            feats = feats.unsqueeze(1)

            output_segmentation, output_spotting = model(feats)

            loss_segmentation = criterion[0](labels, output_segmentation)
            loss_spotting = criterion[1](targets, output_spotting)
            loss = weights[0] * loss_segmentation + weights[1] * loss_spotting

            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
            losses_spotting.update(loss_spotting.item(), feats.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f"Train {epoch}: "
            else:
                desc = f"Evaluate {epoch}: "
            desc += f"Time {batch_time.avg:.3f}s "
            desc += f"(it:{batch_time.val:.3f}s) "
            desc += f"Data:{data_time.avg:.3f}s "
            desc += f"(it:{data_time.val:.3f}s) "
            desc += f"Loss {losses.avg:.4e} "
            desc += f"Loss Seg {losses_segmentation.avg:.4e} "
            desc += f"Loss Spot {losses_spotting.avg:.4e} "
            t.set_description(desc)

    return losses.avg


def test(dataloader, model, model_name, save_predictions=False):
    data_time = AverageMeter()

    spotting_groundtruth_visibility = []
    spotting_predictions = []

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_clips, label) in t:
            data_time.update(time.time() - end)

            feat_clips = feat_clips.cuda().squeeze(0)
            label = label.float().squeeze(0)
            feat_clips = feat_clips.unsqueeze(1)

            output_segmentation, output_spotting = model(feat_clips)

            timestamp_long = timestamps2long(
                output_spotting.cpu().detach(),
                label.size()[0],
                chunk_size,
                receptive_field,
            )
            _ = batch2long(
                output_segmentation.cpu().detach(),
                label.size()[0],
                chunk_size,
                receptive_field,
            )

            spotting_groundtruth_visibility.append(label)
            spotting_predictions.append(timestamp_long)

    targets_numpy = []
    closests_numpy = []
    detections_numpy = []
    for target, detection in zip(spotting_groundtruth_visibility, spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 5 * model.framerate))
        closest_numpy = np.zeros(target_numpy.shape) - 1
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = target_numpy[indexes[i], c]
        closests_numpy.append(closest_numpy)

    if save_predictions:
        list_game = getattr(dataloader.dataset, "match_names", None)
        if list_game is None:
            list_game = [f"match_{idx:04d}" for idx in np.arange(len(dataloader.dataset))]
        inverse_event_dictionary = {
            v: k for k, v in getattr(dataloader.dataset, "dict_event", {}).items()
        }
        for index in np.arange(len(list_game)):
            predictions2json_single(
                detections_numpy[index],
                "outputs/",
                list_game[index],
                model.framerate,
                inverse_event_dictionary=inverse_event_dictionary,
            )

    a_mAP, a_mAP_per_class = average_mAP(
        targets_numpy, detections_numpy, closests_numpy, model.framerate
    )

    print("Average mAP: ", a_mAP)
    print("Average mAP per class: ", a_mAP_per_class)

    return a_mAP, a_mAP_per_class
