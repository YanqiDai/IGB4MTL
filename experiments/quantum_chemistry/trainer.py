from argparse import ArgumentParser
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import trange

import sys
sys.path.append("../..")
from experiments.quantum_chemistry.models import Net
from experiments.quantum_chemistry.utils import (
    Complete,
    MyTransform,
    delta_fn,
    multiply_indx,
)
from experiments.quantum_chemistry.utils import target_idx as targets
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


@torch.no_grad()
def evaluate(model, loader, std, scale_target):
    model.eval()
    data_size = 0.0
    task_losses = 0.0
    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        if scale_target:
            task_losses += F.l1_loss(
                out * std.to(device), data.y * std.to(device), reduction="none"
            ).sum(
                0
            )  # MAE
        else:
            task_losses += F.l1_loss(out, data.y, reduction="none").sum(0)  # MAE
        data_size += len(data.y)

    model.train()

    avg_task_losses = task_losses / data_size

    # Report meV instead of eV.
    avg_task_losses = avg_task_losses.detach().cpu().numpy()
    avg_task_losses[multiply_indx] *= 1000

    delta_m = delta_fn(avg_task_losses)
    return dict(
        avg_loss=avg_task_losses.mean(),
        avg_task_losses=avg_task_losses,
        delta_m=delta_m,
    )


def main(
    data_path: str,
    batch_size: int,
    device: torch.device,
    lr: float,
    n_epochs: int,
    targets: list = None,
    scale_target: bool = True,
    main_task: int = None,
):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{timestr}_{args.loss_method}_{args.gradient_method}_seed{args.seed}_log.txt"

    dim = 64
    model = Net(n_tasks=len(targets), num_features=11, dim=dim).to(device)

    transform = T.Compose([MyTransform(targets), Complete(), T.Distance(norm=False)])
    dataset = QM9(data_path, transform=transform).shuffle()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]

    std = None
    if scale_target:
        mean = train_dataset.data.y[:, targets].mean(dim=0, keepdim=True)
        std = train_dataset.data.y[:, targets].std(dim=0, keepdim=True)

        dataset.data.y[:, targets] = (dataset.data.y[:, targets] - mean) / std

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    loss_weight_method = LossWeightMethods(
        args.loss_method, n_tasks=len(targets), device=device, **loss_weight_methods_parameters[args.loss_method]
    )
    # gradient_weight method
    gradient_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    gradient_weight_method = GradientWeightMethods(
        args.gradient_method, n_tasks=len(targets), device=device, **gradient_weight_methods_parameters[args.gradient_method]
    )

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=loss_weight_method.parameters(), lr=args.method_params_lr),
            dict(params=gradient_weight_method.parameters(), lr=args.method_params_lr),

        ],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
    )

    epoch_iterator = trange(n_epochs)
    train_batch = len(train_loader)

    best_val = np.inf
    best_val_delta = np.inf

    train_time_sum = 0.0

    # reward scale for IGBv2
    if args.loss_method == 'igbv2':
        loss_weight_method.method.train_batch = train_batch

    for epoch in epoch_iterator:
        lr = optimizer.param_groups[0]["lr"]
        avg_train_losses = torch.zeros(len(targets)).to(device)
        avg_loss_weights = torch.zeros(len(targets)).to(device)

        start_train_time = time.time()

        # reward scale for IGBv2
        if args.loss_method == 'igbv2':
            loss_weight_method.method.reward_scale = lr / optimizer.param_groups[0]['lr']

        for j, data in enumerate(train_loader):
            model.train()

            data = data.to(device)
            optimizer.zero_grad()

            out, features = model(data, return_representation=True)

            losses = F.mse_loss(out, data.y, reduction="none").mean(0)
            # print(losses)
            avg_train_losses += losses.detach() / train_batch

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

            epoch_iterator.set_description(
                f"[{epoch} {j + 1}/{train_batch}]"
            )

        # base_losses for IGBv1 and IGBv2
        if 'igb' in args.loss_method and epoch == args.base_epoch:
            loss_weight_method.method.base_losses = avg_train_losses

        end_train_time = time.time()
        train_time_sum += end_train_time - start_train_time

        val_loss_dict = evaluate(model, val_loader, std=std, scale_target=scale_target)
        val_loss = val_loss_dict["avg_loss"]
        val_delta = val_loss_dict["delta_m"]

        results = f"Epoch: {epoch:04d}\n" \
                  f"AVERAGE LOSS WEIGHTS: " \
                  f"{avg_loss_weights[0]:.4f} {avg_loss_weights[1]:.4f} {avg_loss_weights[2]:.4f} " \
                  f"{avg_loss_weights[3]:.4f} {avg_loss_weights[4]:.4f} {avg_loss_weights[5]:.4f} " \
                  f"{avg_loss_weights[6]:.4f} {avg_loss_weights[7]:.4f} {avg_loss_weights[8]:.4f} " \
                  f"{avg_loss_weights[9]:.4f} {avg_loss_weights[10]:.4f}\n" \
                  f"TRAIN: {losses.mean().item():.3f}\n" \
                  f"VAL: {val_loss:.3f} {val_delta:.3f}\n"

        if args.loss_method == "stl":
            best_val_criteria = val_loss_dict["avg_task_losses"][main_task] <= best_val
        else:
            best_val_criteria = val_delta <= best_val_delta

        if best_val_criteria:
            best_val = val_loss
            best_val_delta = val_delta

            test_loss_dict = evaluate(model, test_loader, std=std, scale_target=scale_target)
            test_loss = test_loss_dict["avg_loss"]
            test_task_losses = test_loss_dict["avg_task_losses"]
            test_delta = test_loss_dict["delta_m"]
            test_result = f"TEST: {test_loss:.3f} {test_delta:.3f}\n"
            test_result += f"TEST LOSSES: "
            for i in range(len(targets)):
                test_result += f"{test_task_losses[i]:.3f} "
            test_result = test_result[:-1] + "\n"
            print(test_result, end='')
            results += test_result

        with open(log_file, mode="a") as log_f:
            log_f.write(results)

        scheduler.step(
            val_loss_dict["avg_task_losses"][main_task]
            if args.loss_method == "stl"
            else val_delta
        )

    train_time_log = f"Training time: {int(train_time_sum)}s\n"
    print(train_time_log, end='')
    with open(log_file, mode="a") as log_f:
        log_f.write(train_time_log)


if __name__ == "__main__":
    parser = ArgumentParser("QM9", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset",
        lr=1e-3,
        n_epochs=300,
        batch_size=120,
    )
    parser.add_argument("--scale-y", default=True, type=str2bool)
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        targets=targets,
        scale_target=args.scale_y,
        main_task=args.main_task,
    )
