import torch


def load_crown_dict(model_file):
    checkpoint = torch.load(model_file, map_location='cpu')
    if isinstance(checkpoint["state_dict"], list):
        checkpoint["state_dict"] = checkpoint["state_dict"][0]
    new_state_dict = {}
    for k in checkpoint["state_dict"].keys():
        if "prev" in k:
            pass
        else:
            new_state_dict[k] = checkpoint["state_dict"][k]
    checkpoint["state_dict"] = new_state_dict

    """
    state_dict = m.state_dict()
    state_dict.update(checkpoint["state_dict"])
    m.load_state_dict(state_dict)
    print(checkpoint["state_dict"]["__mask_layer.weight"])
    """

    return checkpoint["state_dict"]
