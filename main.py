import torch
from torch.utils.data import DataLoader
from utils import *
from dataset import SRdataset
from generator import RRDBNet      #
import torch.nn.functional as F


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("устройство:", device) #cuba - видеокарта cpu - проц


    ds = SRdataset("data/TEST/hr", "data/TEST/lr")
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = RRDBNet(num_block=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))


    step = 0
    for epoch in range(50):
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr) #берётся плохое качество
            loss = F.l1_loss(sr, hr) #sr-hr = [3 4 5 ...] находится среднее чем меньше тем лучше

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"эпоха {epoch}, шаг {step}, loss {loss.item()}")

if __name__ == "__main__":
    main()