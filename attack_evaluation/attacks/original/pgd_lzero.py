import numpy as np
from torchattacks import SparseFool

from typing import Optional
from functools import partial

import torch
from torch import Tensor, nn
from foolbox.attacks.ead import EADAttack

from foolbox.attacks.dataset_attack import DatasetAttack
import foolbox as fb

class BinarySearchMinimalWrapperL0:
    def __init__(self, model: nn.Module, attack: partial, init_eps: float, search_steps: int,
                 max_eps: Optional[float] = None, batched: bool = False, device: str = "cpu"):
        self.attack = partial(attack, model=model)
        self.model = model
        self.init_eps = init_eps
        self.search_steps = search_steps
        self.max_eps = max_eps
        self.batched = batched
        self.targeted = False
        self.MAX_ADV_FEATURES = 1000
        self.device = device

    def set_mode_targeted_by_function(self):
        self.attack.set_mode_targeted_by_function()
        self.targeted = True

    def __call__(self, inputs: Tensor, labels: Tensor) -> Tensor:
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        adv_inputs = inputs.clone()
        eps_low = inputs.new_zeros(batch_size, dtype=int).to(self.device)
        best_eps = torch.full_like(eps_low, self.MAX_ADV_FEATURES if self.max_eps is None else 2 * self.max_eps,
                                   dtype=int).to(self.device)
        found_high = torch.full_like(eps_low, False, dtype=torch.bool).to(self.device)

        logits = self.model(inputs)
        preds = logits.argmax(dim=1)
        is_adv = (preds != labels) if self.targeted else (preds != labels)

        eps = torch.full_like(eps_low, self.init_eps, dtype=int).to(self.device)
        for i in range(self.search_steps):
            if self.batched:
                self.attack.eps = batch_view(eps)
                adv_inputs_run = self.attack(inputs=inputs, labels=labels)
            else:
                adv_inputs_run = inputs.clone()
                for eps_ in torch.unique(eps):
                    if eps_.item() > 0:
                        mask = (eps == eps_).to(self.device)
                        eps_i = int(eps_.item())
                        adv_inputs_run[mask] = self.attack(inputs=inputs[mask], labels=labels[mask], epsilon=eps_i)

            logits = self.model(adv_inputs_run)
            preds = logits.argmax(dim=1)
            is_adv = (preds != labels) if self.targeted else (preds != labels)

            better_adv = is_adv & (eps < best_eps)
            adv_inputs[better_adv] = adv_inputs_run[better_adv]

            found_high.logical_or_(better_adv)
            eps_low = torch.where(better_adv, eps_low, eps)
            best_eps = torch.where(better_adv, eps, best_eps)

            eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) // 2, 2 * eps_low)
        return adv_inputs


def get_predictions_and_gradients(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    with torch.enable_grad():
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y.cpu()).numpy()

    return pred, grad


def get_predictions(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)
    model = model.to(device)
    xdev = x.to(device)

    with torch.no_grad():
        output = model(xdev)

    return (output.cpu().max(dim=-1)[1] == y).numpy()


def project_L0_box(y, k, lb, ub):
    ''' projection of the batch y to a batch x such that:
          - each image of the batch x has at most k pixels with non-zero channels
          - lb <= x <= ub '''

    x = np.copy(y)
    p1 = np.sum(x ** 2, axis=-1)
    p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
    p2 = np.sum(p2 ** 2, axis=-1)
    p3 = np.sort(np.reshape(p1 - p2, [p2.shape[0], -1]))[:, -k]
    x = x * (np.logical_and(lb <= x, x <= ub)) + lb * (lb > x) + ub * (x > ub)
    x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)

    return x


def perturb_L0_box(attack, x_nat, y_nat, lb, ub, device):
    ''' PGD attack wrt L0-norm + box constraints

        it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
        such that:
          - each image of the batch adv differs from the corresponding one of
            x_nat in at most k pixels
          - lb <= adv - x_nat <= ub

        it returns also a vector of flags where 1 means no adversarial example found
        (in this case the original image is returned in adv) '''

    if attack.rs:
        x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
        x2 = np.clip(x2, 0, 1)
    else:
        x2 = np.copy(x_nat)

    adv_not_found = np.ones(y_nat.shape)
    adv = np.zeros(x_nat.shape)

    for i in (range(attack.num_steps)):
        if i > 0:
            # pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
            pred, grad = get_predictions_and_gradients(attack.model, x2, y_nat, device)
            adv_not_found = np.minimum(adv_not_found, pred.astype(int))
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= (1e-10 + np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True))
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + attack.step_size * grad,
                        casting='unsafe')

        x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)

    return adv, adv_not_found


class PGDattack():
    def __init__(self, model, args):
        self.model = model
        self.type_attack = args['type_attack']  # 'L0', 'L0+Linf', 'L0+sigma'
        self.num_steps = args['num_steps']  # number of iterations of gradient descent for each restart
        self.step_size = args['step_size']  # step size for gradient descent (\eta in the paper)
        self.n_restarts = args['n_restarts']  # number of random restarts to perform
        self.rs = True  # random starting point
        self.epsilon = args['epsilon']  # for L0+Linf, the bound on the Linf-norm of the perturbation
        self.kappa = args[
            'kappa']  # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks
        self.k = args['sparsity']  # maximum number of pixels that can be modified (k_max in the paper)

    def perturb(self, x_nat, y_nat, device):
        adv = np.copy(x_nat)

        for counter in range(self.n_restarts):
            if counter == 0:
                # corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: x_nat, self.model.y_input: y_nat})
                corr_pred = get_predictions(self.model, x_nat, y_nat, device)
                pgd_adv_acc = np.copy(corr_pred)

            if self.type_attack == 'L0':
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, -x_nat, 1.0 - x_nat, device)

            elif self.type_attack == 'L0+Linf':
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.epsilon, -x_nat),
                                                               np.minimum(self.epsilon, 1.0 - x_nat))

            # elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 3:
            #    x_batch_adv, curr_pgd_adv_acc = perturb_L0_sigma(self, x_nat, y_nat)

            elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 1:
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat,
                                                               np.maximum(-self.kappa * self.sigma, -x_nat),
                                                               np.minimum(self.kappa * self.sigma, 1.0 - x_nat))

            pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
            adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
        return adv, pgd_adv_acc


def PGD0(model, inputs, labels, epsilon, attack_args):
    attack_args['sparsity'] = epsilon
    attack = PGDattack(model, attack_args)
    device = inputs.get_device()
    x_test = inputs.permute(0, 2, 3, 1).cpu().numpy()
    y_test = labels.cpu().numpy()
    adv, _ = attack.perturb(x_test, y_test, device)
    x_torch = torch.from_numpy(adv).permute(0, 3, 1, 2).to(device)
    return x_torch


def PGD0_minimal(model: torch.nn.Module, inputs: Tensor, labels: Tensor, search_steps: int = 10, num_steps: int = 100, step_size: float = 120000.0 /255.0,
                kappa: float = -1, epsilon: float = -1, init_eps:int = 100, n_restarts: int = 1):
    attack_args = {
        'type_attack': 'L0',
        'n_restarts': n_restarts,
        'num_steps': num_steps,
        'step_size': step_size,
        'kappa': kappa,
        'epsilon': epsilon,
    }
    pgd0 = partial(PGD0, attack_args=attack_args)
    pgd_minimal = BinarySearchMinimalWrapperL0(model=model, attack=pgd0, init_eps=init_eps, search_steps=search_steps,
                                               device=inputs.get_device())
    adv_examples = pgd_minimal(inputs=inputs, labels=labels)
    return adv_examples

