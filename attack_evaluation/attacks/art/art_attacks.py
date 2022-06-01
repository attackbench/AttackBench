from typing import Optional, Callable
from torch import Tensor, nn, from_numpy
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod, SaliencyMapMethod, CarliniL2Method, \
    CarliniLInfMethod, BrendelBethgeAttack, DeepFool


def art_lib_pgd(model, targeted, batch_size, **kwargs):
    return ProjectedGradientDescent(estimator=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_fgsm(model, targeted, batch_size, **kwargs):
    return FastGradientMethod(estimator=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_jsma(model, targeted, batch_size, **kwargs):
    return SaliencyMapMethod(classifier=model, batch_size=batch_size, **kwargs)


def art_lib_cw_l2(model, targeted, batch_size, **kwargs):
    return CarliniL2Method(classifier=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_cw_linf(model, targeted, batch_size, **kwargs):
    return CarliniLInfMethod(classifier=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_bb(model, targeted, batch_size, **kwargs):
    return BrendelBethgeAttack(estimator=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_deepfool(model, targeted, batch_size, **kwargs):
    return DeepFool(classifier=model, batch_size=batch_size, **kwargs)


def art_lib_wrapper(attack: Callable,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False,
                    **kwargs) -> Tensor:
    loss = nn.CrossEntropyLoss()
    input_shape = inputs.shape[1:]
    output_shape = [module for module in model.modules()][-1].out_features
    art_model = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss,
                                  input_shape=input_shape, nb_classes=output_shape)
    batch_size = inputs.shape[0]
    attack = attack(
        model=art_model, batch_size=batch_size, targeted=targeted, **kwargs)
    y = targets if targeted else labels
    adv_inputs = attack.generate(x=inputs.cpu().numpy(), y=y.cpu().numpy())
    return from_numpy(adv_inputs).to(inputs.device)
