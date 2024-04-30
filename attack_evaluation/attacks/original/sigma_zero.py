from typing import Optional

import torch
import torch.optim.lr_scheduler as lr_scheduler
from adv_lib.utils.losses import difference_of_logits
from foolbox import PyTorchModel
from foolbox.attacks.dataset_attack import DatasetAttack
from torch import Tensor


def l0_mid_points(
        x0: Tensor,
        x1: Tensor,
        ε: Tensor,
        bounds,
):
    # returns a point between x0 and x1 where
    # epsilon = 0 returns x0 and epsilon = 1
    # returns x1

    # get ε in right shape for broadcasting
    ε = ε.reshape(ε.shape + (1,) * (x0.ndim - 1))

    threshold = (bounds[1] - bounds[0]) * ε
    mask = (x1 - x0).abs() < threshold
    new_x = torch.where(mask, x1, x0)
    return new_x


def delta_init(model, inputs, labels, device, starting_point=None, binary_search_steps=10, targeted=False):
    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

    if starting_point is None:
        delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    elif starting_point == 'adversarial':
        fmodel = PyTorchModel(model, bounds=(0, 1))
        dataset_atk = DatasetAttack()

        dataset_atk.feed(fmodel, inputs)
        _, starting_points, success = dataset_atk(fmodel, inputs, labels, epsilons=None)
        is_adv = model(starting_points).argmax(dim=1) != labels
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device)
        for _ in range(binary_search_steps):
            ε = (lower_bound + upper_bound) / 2
            mid_points = l0_mid_points(x0=inputs, x1=starting_points, ε=ε, bounds=[0, 1])
            mid_points = mid_points.reshape(inputs.shape)
            pred_labels = model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, ε)
            upper_bound = torch.where(is_adv, ε, upper_bound)

        mid_points = l0_mid_points(x0=inputs, x1=starting_points, ε=upper_bound, bounds=[0, 1])
        mid_points = mid_points.reshape(inputs.shape)
        delta = mid_points - inputs
        delta.requires_grad_()
    else:
        raise "Not implemented error, only None or 'adversarial' are considered"

    return delta


def sigma_zero(model: torch.nn.Module,
               inputs: Tensor,
               labels: Tensor,
               steps: int = 1000,
               lr: float = 1.0,
               sigma: float = 1e-3,
               thr_0: float = 0.3,
               thr_lr: float = 0.01,
               verbose: bool = False,
               starting_point=None,
               binary_search_steps: int = 10,
               targets: Optional[Tensor] = None,
               targeted: bool = False,
               grad_norm=torch.inf) -> Tensor:
    clamp = lambda tensor: tensor.data.add_(inputs.data).clamp_(min=0, max=1).sub_(inputs.data)
    l0_approximation = lambda tensor, sigma: tensor.square().div(tensor.square().add(sigma)).sum(dim=1)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    normalize = lambda tensor: (
            tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)).view(
        tensor.shape)

    device = next(model.parameters()).device
    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

    delta = delta_init(model, inputs, labels, device, starting_point=starting_point,
                       binary_search_steps=binary_search_steps)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)
    best_l0 = torch.full((batch_size,), max_size, device=device)
    best_delta = delta.clone()

    # th = threshold * torch.ones(size=(batch_size,))
    th = torch.ones(size=inputs.shape, device=device) * thr_0

    for i in range(steps):
        adv_inputs = inputs + delta

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, labels).clip(0)
        l0_approx = l0_approximation(delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / delta.data.flatten(1).shape[1]
        # keep best solutions
        predicted_classes = (logits).argmax(1)
        true_l0 = delta.data.flatten(1).ne(0).sum(dim=1)
        is_not_adv = (predicted_classes == labels)
        is_adv = (predicted_classes != labels)
        is_smaller = (true_l0 < best_l0)
        is_both = is_adv & is_smaller
        best_l0 = torch.where(is_both, true_l0.detach(), best_l0)
        best_delta = torch.where(batch_view(is_both), delta.data.clone().detach(), best_delta)

        # update step
        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

        if verbose and i % 100 == 0:
            print(th.flatten(1).mean(dim=1), th.flatten(1).mean(dim=1).shape)
            print(is_not_adv)
            print(
                f"iter: {i}, dl loss: {dl_loss.mean().item():.4f}, l0 normalized loss: {l0_approx_normalized.mean().item():.4f}, current median norm: {delta.data.flatten(1).ne(0).sum(dim=1).median()}")

        optimizer.zero_grad()
        adv_loss.backward()
        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()
        clamp(delta.data)

        # dynamic thresholding
        th[is_not_adv, :, :, :] -= thr_lr * scheduler.get_last_lr()[0]
        th[~is_not_adv, :, :, :] += thr_lr * scheduler.get_last_lr()[0]
        th.clamp_(0, 1)

        # filter components
        delta.data[delta.data.abs() < th] = 0

    return (inputs + best_delta)
