import torch
from tqdm import tqdm
from model import Modle
from reader import Reader
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
train_reader=Reader("./data/SST-cla/train.txt",0)
test_reader=Reader("./data/SST-cla/train.txt",train_reader.max_line)
train_loader=torch.utils.data.DataLoader(train_reader,shuffle=True,batch_size=64)
test_loader=torch.utils.data.DataLoader(test_reader,shuffle=True,batch_size=64)
model=Modle(train_reader.max_line,5)
model=model.cuda()
model.train()
loss=[]
learning_epoch=10
for i in range(learning_epoch):
    losss=[]
    acc=0
    total=0
    for num,(score,data) in tqdm(enumerate(train_loader)):
        score=score.cuda()
        data=data.cuda()
        predict=model(data)
        loss=torch.nn.CrossEntropyLoss(predict,score)
        losss.append(loss.item())
        acc+=(score==torch.argmax(predict,-1)).cpu().sum().item()
        total+=score.size(1)
    print("training epoch %s accuracy is %s loss is %s "%(str(i),str(acc/total),str(np.mean(losss))))
    losss = []
    acc = 0
    total = 0
    for num, (score, data) in tqdm(enumerate(test_loader)):
        score = score.cuda()
        data = data.cuda()
        predict = model(data)
        loss = torch.nn.CrossEntropyLoss(predict, score)
        losss.append(loss.item())
        acc += (score == torch.argmax(predict, -1)).cpu().sum().item()
        total += score.size(1)
    print("test epoch %s accuracy is %s loss is %s " % (str(i), str(acc / total), str(np.mean(losss))))
