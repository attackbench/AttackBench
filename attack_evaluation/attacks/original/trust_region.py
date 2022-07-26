# Adapted from https://github.com/amirgholami/TRAttack/blob/master/trattack/attack_methods.py
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
#################################################
## Select TR Attack Index
#################################################
from torch import Tensor, nn


def select_index(model, data, c=9, p=2, worst_case=False):
    '''
    Select the attack target class
    '''
    model.eval()
    # COMMENTED: data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    output, ind = torch.sort(output, descending=True)
    n = len(data)
    q = 2
    if p == 8:
        q = 1  # We need to use conjugate norm

    true_out = output[range(n), n * [0]]
    # Backpropogate a batch of images
    z_true = torch.sum(true_out)
    data.grad = None
    z_true.backward(retain_graph=True)
    true_grad = data.grad
    pers = torch.zeros(len(data), 1 + c).cuda()

    for i in range(1, 1 + c):
        z = torch.sum(output[:, i])
        data.grad = None
        model.zero_grad()
        z.backward(retain_graph=True)
        grad = data.grad  # dzi/dx
        grad_diff = torch.norm(grad.data.view(n, -1) - true_grad.data.view(n, -1), p=q, dim=1)  # batch_size x 1
        pers[:, i] = (true_out.data - output[:, i].data) / grad_diff  # batch_size x 1

    if not worst_case:
        pers[range(n), n * [0]] = np.inf
        pers[pers < 0] = 0
        per, index = torch.min(pers, 1)  # batch_size x 1
    else:
        pers[range(n), n * [0]] = -np.inf
        per, index = torch.max(pers, 1)  # batch_size x 1

    output = []
    for i in range(data.size(0)):
        output.append(ind[i, index[i]].item())
    return torch.from_numpy(np.array(output)).to(data.device)  # MODIFIED: return np.array(output)


#################################################
## TR First Order Attack
#################################################
def _tr_attack(model, data, true_ind, target_ind, eps, p=2):
    """Generate an adversarial pertubation using the TR method.
    Pick the top false label and perturb towards that.
    First-order attack

    Args:
        data: input image to perturb
        true_ind: is true label
        target_ind: is the attack label
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    n = len(data)

    q = 2
    if p == 8:
        q = 1

    output_g = output[range(n), target_ind] - output[range(n), true_ind]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = deepcopy(data.grad.data)
    update = update.view(n, -1)
    per = (-output_g.data.view(n, -1) + 0.) / (torch.norm(update, p=q, dim=1).view(n, 1) + 1e-6)

    if p == 8:
        update = torch.sign(update)
    elif p == 2:
        update = update.view(n, -1)
        update = update / (torch.norm(update, p=2, dim=1).view(n, 1) + 1e-6)
    per = per.view(-1)
    per_mask = per > eps
    per_mask = per_mask.nonzero().view(-1)
    # set overshoot for small pers
    per[per_mask] = eps
    X_adv = data.data + (((per + 1e-4) * 1.02).view(n, -1) * update.view(n, -1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv


def tr_attack_iter(model, data, target, eps, c=9, p=2, iter=100, worst_case=False):
    X_adv = data.clone()  # MODIFIED: X_adv = deepcopy(data.cuda())
    target_ind = select_index(model, data, c=c, p=p, worst_case=worst_case)

    update_num = 0.
    for i in range(iter):
        model.eval()
        Xdata, Ytarget = X_adv, target  # MODIFIED: Xdata, Ytarget = X_adv, target.cuda()
        # First check if the input is correctly classfied before attack
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Ytarget) == Ytarget.data  # get index
        update_num += torch.sum(tmp_mask.long())
        # if all images are incorrectly classfied the attack is successful and exit
        if torch.sum(tmp_mask.long()) < 1:
            return X_adv, update_num  # MODIFIED: return X_adv.cpu(), update_num
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask, :] = _tr_attack(model, X_adv[attack_mask, :], target[attack_mask], target_ind[attack_mask],
                                           eps, p=p)
    return X_adv, update_num  # MODIFIED: return X_adv.cpu(), update_num


#################################################
## TR First Order Attack Adaptive
#################################################
def _tr_attack_adaptive(model, data, true_ind, target_ind, eps, p=2):
    """Generate an adversarial pertubation using the TR method with adaptive
    trust radius.
    Args:
        data: input image to perturb
        true_ind: is true label
        target_ind: is the attack label
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    n = len(data)

    q = 2
    if p == 8:
        q = 1

    output_g = output[range(n), target_ind] - output[range(n), true_ind]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = deepcopy(data.grad.data)
    update = update.view(n, -1)
    per = (-output_g.data.view(n, -1) + 0.) / (torch.norm(update, p=q, dim=1).view(n, 1) + 1e-6)

    if p == 8:
        update = torch.sign(update)
    elif p == 2:
        update = update.view(n, -1)
        update = update / (torch.norm(update, p=2, dim=1).view(n, 1) + 1e-6)

    ### set large per to eps
    per = per.view(-1)
    eps = eps.view(-1)
    per_mask = per > eps
    per_mask = per_mask.nonzero().view(-1)
    # set overshoot for small pers
    per[per_mask] = eps[per_mask]
    per = per.view(n, -1)
    eps = deepcopy(per)
    X_adv = data.data + (1.02 * (eps + 1e-4) * update.view(n, -1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))

    ### update eps magnitude
    ori_diff = -output_g.data + 0.0

    adv_output = model(X_adv)
    adv_diff = adv_output[range(n), true_ind] - output[range(n), target_ind]

    eps = eps.view(-1)
    obj_diff = (ori_diff - adv_diff) / eps

    increase_ind = obj_diff > 0.9
    increase_ind = increase_ind.nonzero().view(-1)

    decrease_ind = obj_diff < 0.5
    decrease_ind = decrease_ind.nonzero().view(-1)

    eps[increase_ind] = eps[increase_ind] * 1.2
    eps[decrease_ind] = eps[decrease_ind] / 1.2

    if p == 2:
        eps_max = 0.05
        eps_min = 0.0005
        eps_mask = eps > eps_max
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_max
        eps_mask = eps < eps_min
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_min

    elif p == 8:
        eps_max = 0.01
        eps_min = 0.0001
        eps_mask = eps > eps_max
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_max
        eps_mask = eps < eps_min
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_min

    eps = eps.view(n, -1)
    return X_adv, eps


def tr_attack_adaptive_iter(model, data, target, eps, c=9, p=2, iter=100, worst_case=False):
    X_adv = data.clone()  # MODIFIED: X_adv = deepcopy(data.cuda())
    target_ind = select_index(model, data, c=c, p=p, worst_case=worst_case)

    update_num = 0.
    eps = torch.from_numpy(np.array([eps] * len(data))).view(len(data), -1)
    eps = eps.float().to(data.device)  # MODIFIED: eps = eps.type(torch.FloatTensor).cuda()
    for i in range(iter):
        model.eval()
        Xdata, Ytarget = X_adv, target  # MODIFIED: Xdata, Ytarget = X_adv, target.cuda()
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Ytarget) == Ytarget.data  # get index
        update_num += torch.sum(tmp_mask.long())
        if torch.sum(tmp_mask.long()) < 1:
            return X_adv, update_num  # MODIFIED: return X_adv.cpu(), update_num
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask, :], eps[attack_mask, :] = _tr_attack_adaptive(model, X_adv[attack_mask, :],
                                                                         target[attack_mask], target_ind[attack_mask],
                                                                         eps[attack_mask, :], p=p)
    return X_adv, update_num  # MODIFIED: return X_adv.cpu(), update_num


_norms = {
    'l2': 2,
    'linf': 8,
}


def tr_attack(model: nn.Module,
              inputs: Tensor,
              labels: Tensor,
              threat_model: str,
              targets: Optional[Tensor] = None,
              targeted: bool = False,
              adaptive: bool = False,
              eps: float = 0.001,
              c: int = 9,
              iter: int = 100) -> Tensor:
    attack_func = tr_attack_adaptive_iter if adaptive else tr_attack_iter
    adv_inputs = attack_func(model=model, data=inputs, target=labels, p=_norms[threat_model], eps=eps, c=c, iter=iter)
    return adv_inputs[0]
