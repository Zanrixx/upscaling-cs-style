import os
import torch


def save_checkpoint(path, model, optimizer, step):
    state = {
        "model": model.state_dict(),          #веса сети
        "optimizer": optimizer.state_dict(),  #служебные данные Adam
        "step": step,                        
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)   
    os.replace(tmp, path)     


def load_checkpoint(path, model, optimizer, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


