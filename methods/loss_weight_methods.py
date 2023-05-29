from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from methods.weight_method import WeightMethod, LinearScalarization
from methods.SAC_Agent import SAC_Agent, RandomBuffer


class ScaleInvariantLinearScalarization(WeightMethod):
    """Scale-invariant loss balancing paradigm"""

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        losses = torch.log(losses) * self.task_weights
        return losses, dict(weights=self.task_weights)


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        losses = losses * self.weights
        return losses, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(losses / (2 * self.logsigma.exp()) + self.logsigma / 2)
        return loss, dict(weights=torch.exp(-self.logsigma))  # NOTE: not exactly task weights

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        losses = losses / (2 * self.logsigma.exp()) + self.logsigma / 2
        return losses, dict(weights=torch.exp(-self.logsigma))

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class UncertaintyLog(WeightMethod):
    """UW + SI"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(torch.log(losses) / (2 * self.logsigma.exp()) + self.logsigma / 2)
        return loss, dict(weights=torch.exp(-self.logsigma))  # NOTE: not exactly task weights

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        losses = torch.log(losses) / (2 * self.logsigma.exp()) + self.logsigma / 2
        return losses, dict(weights=torch.exp(-self.logsigma))  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        weight = (self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        losses = losses * weight
        return losses, dict(weights=weight)


class RLWLog(WeightMethod):
    """RLW + SI"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(torch.log(losses) * weight)

        return loss, dict(weights=weight)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        weight = (self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        losses = torch.log(losses) * weight
        return losses, dict(weights=weight)


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
            self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = sum(task_weights * losses)

        self.running_iterations += 1

        return loss, dict(weights=task_weights)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        losses = task_weights * losses

        self.running_iterations += 1

        return losses, dict(weights=task_weights)


class DynamicWeightAverageLog(WeightMethod):
    """DWA + SI"""
    def __init__(
            self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = sum(task_weights * torch.log(losses))

        self.running_iterations += 1

        return loss, dict(weights=task_weights)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        losses = task_weights * torch.log(losses)

        self.running_iterations += 1

        return losses, dict(weights=task_weights)


class ImprovableGapBalancing_v1(WeightMethod):
    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.base_losses = None
        self.weights = torch.ones(n_tasks).to(device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        if self.base_losses is not None:
            self.weights = self.n_tasks * F.softmax(losses.detach() / self.base_losses, dim=-1).to(losses.device)
        loss = sum(self.weights * torch.log(losses))
        return loss, dict(weights=self.weights)  # NOTE: not exactly task weights

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        if self.base_losses is not None:
            self.weights = self.n_tasks * F.softmax(losses.detach() / self.base_losses, dim=-1).to(losses.device)
        losses = self.weights * torch.log(losses)
        return losses, dict(weights=self.weights)  # NOTE: not exactly task weights


class ImprovableGapBalancing_v2(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, sac_lr=3e-4, buffer_size=1e4):
        super().__init__(n_tasks, device=device)
        self.base_losses = None
        self.weights = torch.ones(n_tasks).to(device)

        self.sac_model = SAC_Agent(state_dim=n_tasks, action_dim=n_tasks, a_lr=sac_lr, c_lr=sac_lr, batch_size=256, device=device)
        self.replay_buffer = RandomBuffer(state_dim=n_tasks, action_dim=n_tasks, max_size=buffer_size, device=device)
        self.custom_step = 0
        self.bool_custom_step = 0
        self.batch_loss = torch.zeros([2, n_tasks]).to(device)
        self.batch_rl_weight = torch.zeros([2, n_tasks]).to(device)
        self.train_batch = None
        self.start_epoch = 5
        self.update_after = 3
        self.update_every = 50
        self.reward_scale = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        self.batch_loss[self.bool_custom_step] = losses.detach()

        # write random buffer
        if self.base_losses is not None:
            loss_de = (self.batch_loss[(self.bool_custom_step - 1) % 2] - self.batch_loss[self.bool_custom_step])
            loss_de = loss_de / self.base_losses
            reward = min(loss_de)
            reward *= self.reward_scale
            self.replay_buffer.add(self.batch_loss[(self.bool_custom_step - 1) % 2],
                                   self.batch_rl_weight[(self.bool_custom_step - 1) % 2],
                                   reward,
                                   self.batch_loss[self.bool_custom_step])
        # train sac_model
        if self.custom_step >= self.update_after * self.train_batch and self.custom_step % self.update_every == 0:
            k = 1 + self.replay_buffer.size / self.replay_buffer.max_size
            for i in range(int(self.update_every * (k / 2))):
                self.sac_model.train(self.replay_buffer, k)
        # change weights
        if self.custom_step < self.start_epoch * self.train_batch:
            self.weights = self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1).to(self.device)
        else:
            self.weights = self.sac_model.select_action(self.batch_loss[self.bool_custom_step],
                                                        deterministic=False,
                                                        with_logprob=False)
        self.batch_rl_weight[self.bool_custom_step] = self.weights.detach()

        loss = sum(self.weights * torch.log(losses))

        self.custom_step += 1
        self.bool_custom_step = (self.bool_custom_step + 1) % 2

        return loss, dict(weights=self.weights)     # NOTE: not exactly task weights

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        self.batch_loss[self.bool_custom_step] = losses.detach()

        # write random buffer
        if self.base_losses is not None:
            loss_de = (self.batch_loss[(self.bool_custom_step - 1) % 2] - self.batch_loss[self.bool_custom_step])
            loss_de = loss_de / self.base_losses
            reward = min(loss_de)
            # reward = sum(loss_de) / self.n_tasks
            reward *= self.reward_scale
            self.replay_buffer.add(self.batch_loss[(self.bool_custom_step - 1) % 2],
                                   self.batch_rl_weight[(self.bool_custom_step - 1) % 2],
                                   reward,
                                   self.batch_loss[self.bool_custom_step])
        # train sac_model
        if self.custom_step >= self.update_after * self.train_batch and self.custom_step % self.update_every == 0:
            k = 1 + self.replay_buffer.size / self.replay_buffer.max_size
            for i in range(int(self.update_every * (k / 2))):
                self.sac_model.train(self.replay_buffer, k)
        # give weights
        if self.custom_step < self.start_epoch * self.train_batch:
            self.weights = self.n_tasks * F.softmax(torch.randn(self.n_tasks), dim=-1).to(self.device)
        else:
            self.weights = self.sac_model.select_action(self.batch_loss[self.bool_custom_step],
                                                        deterministic=False,
                                                        with_logprob=False)
        self.batch_rl_weight[self.bool_custom_step] = self.weights.detach()

        losses = self.weights * torch.log(losses)

        self.custom_step += 1
        self.bool_custom_step = (self.bool_custom_step + 1) % 2

        return losses, dict(weights=self.weights)   # NOTE: not exactly task weights


class LossWeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(LOSS_METHODS.keys()), f"unknown method {method}."

        self.method = LOSS_METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def get_weighted_losses(self, losses: torch.Tensor, **kwargs):
        return self.method.get_weighted_losses(losses, **kwargs)

    def backward(
            self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


LOSS_METHODS = dict(
    ls=LinearScalarization,
    stl=STL,
    si=ScaleInvariantLinearScalarization,
    uw=Uncertainty,
    uwlog=UncertaintyLog,
    rlw=RLW,
    rlwlog=RLWLog,
    dwa=DynamicWeightAverage,
    dwalog=DynamicWeightAverageLog,
    igbv1=ImprovableGapBalancing_v1,
    igbv2=ImprovableGapBalancing_v2,
)
