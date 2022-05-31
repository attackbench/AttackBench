from typing import Optional
from torch import Tensor, nn, from_numpy
from art.estimators.classification import PyTorchClassifier
from art.attacks.attack import EvasionAttack


def art_lib_wrapper(attack: EvasionAttack,
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
        estimator=art_model, batch_size=batch_size, targeted=targeted, **kwargs)
    y = targets if targeted else labels
    adv_inputs = attack.generate(x=inputs.cpu().numpy(), y=y.cpu().numpy())
    return from_numpy(adv_inputs).to(inputs.device)
