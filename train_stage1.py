#1 стадия будет только генератор, потеря l1 и выбор лучшей точки по PSNR
import torch
from torch.utils.data import DataLoader
from utils import *
from dataset import SRdataset
from generator import RRDBNet      #
import torch.nn.functional as F

def train_stage1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("видеокарта не найдена")
        return 
    ds = SRdataset("data/TEST/hr", "data/TEST/lr")