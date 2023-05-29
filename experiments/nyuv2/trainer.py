import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

import sys
sys.path.append("../..")
from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error, stl_eval_mean
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)

from methods.loss_weight_methods import LossWeightMethods
from methods.gradient_weight_methods import GradientWeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(path, lr, bs, device):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{timestr}_{args.loss_method}_{args.gradient_method}_seed{args.seed}_log.txt"

    # Nets
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(root=path.as_posix(), mode="train", augmentation=args.apply_augmentation)
    nyuv2_val_set = NYUv2(root=path.as_posix(), mode="val")
    nyuv2_test_set = NYUv2(root=path.as_posix(), mode="test")

    train_loader = DataLoader(dataset=nyuv2_train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=nyuv2_val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(dataset=nyuv2_test_set, batch_size=bs, shuffle=False)

    # loss_weight method
    loss_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    loss_weight_method = LossWeightMethods(
        args.loss_method, n_tasks=3, device=device, **loss_weight_methods_parameters[args.loss_method]
    )

    # gradient_weight method
    gradient_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    gradient_weight_method = GradientWeightMethods(
        args.gradient_method, n_tasks=3, device=device, **gradient_weight_methods_parameters[args.gradient_method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=loss_weight_method.parameters(), lr=args.method_params_lr),
            dict(params=gradient_weight_method.parameters(), lr=args.method_params_lr),
        ],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    val_batch = len(val_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)

    # best model to test
    best_epoch = None
    best_eval = 0

    # print result head
    print(
        f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
        f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m"
    )

    train_time_sum = 0.0

    # train batch for IGBv2
    if args.loss_method == 'igbv2':
        loss_weight_method.method.train_batch = train_batch

    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)
        conf_mat = ConfMatrix(model.segnet.class_nb)
        avg_loss_weights = torch.zeros(3).to(device)

        start_train_time = time.time()

        # reward scale for IGBv2
        if args.loss_method == 'igbv2':
            loss_weight_method.method.reward_scale = lr / optimizer.param_groups[0]['lr']

        for j, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack((calc_loss(train_pred[0], train_label, "semantic"),
                                  calc_loss(train_pred[1], train_depth, "depth"),
                                  calc_loss(train_pred[2], train_normal, "normal")))

            weighted_losses, loss_weights = loss_weight_method.get_weighted_losses(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )
            avg_loss_weights += loss_weights['weights'] / train_batch

            loss, gradient_weights = gradient_weight_method.backward(
                losses=weighted_losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch} {j + 1}/{train_batch}] losses: {losses[0].item():.3f} "
                f"{losses[1].item():.3f} {losses[2].item():.3f} "
                f"weights: {loss_weights['weights'][0].item():.3f} "
                f"{loss_weights['weights'][1].item():.3f} {loss_weights['weights'][2].item():.3f}"
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # base_losses for IGBv1 and IGBv2
        if 'igb' in args.loss_method and epoch == args.base_epoch:
            base_losses = torch.Tensor(avg_cost[epoch, [0, 3, 6]]).to(device)
            loss_weight_method.method.base_losses = base_losses

        end_train_time = time.time()
        train_time_sum += end_train_time - start_train_time

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            for j, batch in enumerate(val_loader):
                val_data, val_label, val_depth, val_normal = batch
                val_data, val_label = val_data.to(device), val_label.long().to(device)
                val_depth, val_normal = val_depth.to(device), val_normal.to(device)

                val_pred = model(val_data)
                val_loss = torch.stack(
                    (
                        calc_loss(val_pred[0], val_label, "semantic"),
                        calc_loss(val_pred[1], val_depth, "depth"),
                        calc_loss(val_pred[2], val_normal, "normal"),
                    )
                )

                conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

                cost[12] = val_loss[0].item()
                cost[15] = val_loss[1].item()
                cost[16], cost[17] = depth_error(val_pred[1], val_depth)
                cost[18] = val_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred[2], val_normal)
                avg_cost[epoch, 12:] += cost[12:] / val_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Val Delta_m
            val_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )

        if args.loss_method != "stl":
            eval_value = val_delta_m
        else:
            eval_value = stl_eval_mean(avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]], args.main_task)

        results = f"Epoch: {epoch:04d}\n" \
                  f"AVERAGE LOSS WEIGHTS: " \
                  f"{avg_loss_weights[0]:.4f} {avg_loss_weights[1]:.4f} {avg_loss_weights[2]:.4f}\n" \
                  f"TRAIN: " \
                  f"{avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} | " \
                  f"{avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | " \
                  f"{avg_cost[epoch, 6]:.4f} {avg_cost[epoch, 7]:.2f} {avg_cost[epoch, 8]:.2f} " \
                  f"{avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}\n" \
                  f"VAL: " \
                  f"{avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | " \
                  f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | " \
                  f"{avg_cost[epoch, 18]:.4f} {avg_cost[epoch, 19]:.2f} {avg_cost[epoch, 20]:.2f} " \
                  f"{avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} | " \
                  f"{val_delta_m:.3f}\n"

        if best_epoch is None or eval_value < best_eval:
            best_epoch = epoch
            best_eval = eval_value

            # test
            test_cost = np.zeros(12, dtype=np.float32)
            test_avg_cost = np.zeros(12, dtype=np.float32)
            conf_mat = ConfMatrix(model.segnet.class_nb)
            with torch.no_grad():
                for j, batch in enumerate(test_loader):
                    test_data, test_label, test_depth, test_normal = batch
                    test_data, test_label = test_data.to(device), test_label.long().to(device)
                    test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                    test_pred = model(test_data)
                    test_loss = torch.stack(
                        (
                            calc_loss(test_pred[0], test_label, "semantic"),
                            calc_loss(test_pred[1], test_depth, "depth"),
                            calc_loss(test_pred[2], test_normal, "normal"),
                        )
                    )

                    conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                    test_cost[0] = test_loss[0].item()
                    test_cost[3] = test_loss[1].item()
                    test_cost[4], test_cost[5] = depth_error(test_pred[1], test_depth)
                    test_cost[6] = test_loss[2].item()
                    test_cost[7], test_cost[8], test_cost[9], test_cost[10], test_cost[11] = normal_error(
                        test_pred[2], test_normal
                    )
                    test_avg_cost += test_cost / test_batch

                # compute mIoU and acc
                test_avg_cost[1:3] = conf_mat.get_metrics()

                # Test Delta_m
                test_delta_m = delta_fn(
                    test_avg_cost[[1, 2, 4, 5, 7, 8, 9, 10, 11]]
                )
            test_result = f"TEST: {test_avg_cost[0]:.4f} {test_avg_cost[1]:.4f} {test_avg_cost[2]:.4f} | " \
                          f"{test_avg_cost[3]:.4f} {test_avg_cost[4]:.4f} {test_avg_cost[5]:.4f} | " \
                          f"{test_avg_cost[6]:.4f} {test_avg_cost[7]:.2f} {test_avg_cost[8]:.2f} " \
                          f"{test_avg_cost[9]:.4f} {test_avg_cost[10]:.4f} {test_avg_cost[11]:.4f} | " \
                          f"{test_delta_m:.3f}\n"
            results += test_result
            # print test result
            print(test_result, end='')
        with open(log_file, mode="a") as log_f:
            log_f.write(results)

    train_time_log = f"Training time: {int(train_time_sum)}s\n"
    print(train_time_log, end='')
    with open(log_file, mode="a") as log_f:
        log_f.write(train_time_log)


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset",
        lr=1e-4,
        n_epochs=500,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="segnet",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation",
        type=str2bool,
        default=True,
        help="data augmentations"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)
