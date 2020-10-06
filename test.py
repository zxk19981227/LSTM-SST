import torch
from torchsummaryX import summary
from tqdm import tqdm
import numpy as np
print("reading glove")
weight=[]
with open("./data/SST-cla/glove.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = line.strip().split(' ')[1:]
        if data==[]:
            continue
        sen = [float(i) for i in data]
        weight.append(sen)
        print(len(sen))
        assert len(sen) == 300
weights=np.array(weight)