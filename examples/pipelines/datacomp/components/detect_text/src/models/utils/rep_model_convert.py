import copy
import torch


def rep_model_convert(model: torch.nn.Module, save_path=None, do_copy=False):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
            # print("switch_to_deploy")
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
