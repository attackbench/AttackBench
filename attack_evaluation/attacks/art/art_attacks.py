from typing import Optional, Callable
from torch import Tensor, nn, from_numpy
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod, SaliencyMapMethod


def art_lib_pgd(model, targeted, batch_size, **kwargs):
    return ProjectedGradientDescent(estimator=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_fgsm(model, targeted, batch_size, **kwargs):
    return FastGradientMethod(estimator=model, targeted=targeted, batch_size=batch_size, **kwargs)


def art_lib_jsma(model, targeted, batch_size, **kwargs):
    return SaliencyMapMethod(classifier=model, batch_size=batch_size, **kwargs)


_art_attacks = {
    'pgd': art_lib_pgd,
    'fgsm': art_lib_fgsm,
    'jsma': art_lib_jsma,
}


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
