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

def convert_keys(w):
    new = {}                       
    for k, v in w.items():          
        k = k.replace("RRDB_trunk.", "body.")
        k = k.replace(".RDB1.", ".rdb1.").replace(".RDB2.", ".rdb2.").replace(".RDB3.", ".rdb3.")
        k = k.replace("trunk_conv", "conv_body")
        k = k.replace("upconv1", "conv_up1").replace("upconv2", "conv_up2")
        k = k.replace("HRconv", "conv_hr")
        new[k] = v                  
    return new
