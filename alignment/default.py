import os
import torch
from alignment.deform_net import DeformNet
import options


def load_default_nnrt_network():
    saved_model = options.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct alignment
    model = DeformNet().cuda()

    if "chairs_things" in saved_model:
        model.flow_net.load_state_dict(pretrained_dict)
    else:
        if options.model_module_to_load == "full_model":
            # Load completely alignment
            model.load_state_dict(pretrained_dict)
        elif options.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            print(options.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    model.eval()
    return model
