import torch
from tqdm import tqdm
from torch.autograd import  Variable
from T_f import Modle
from reader import Reader
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("create train reader")
train_reader=Reader("./data/SST-cla/train.txt",0)
print("create test reader")
test_reader=Reader("./data/SST-cla/dev.txt",train_reader.max_line)
print("max sentence length",train_reader.max_line)
train_loader=torch.utils.data.DataLoader(train_reader,shuffle=True,batch_size=256)
test_loader=torch.utils.data.DataLoader(test_reader,shuffle=False,batch_size=256)
print("building model")
model=Modle(train_reader.max_line,5,1,10,0.5)
print("model finished")
model=model.cuda()
model.train()
loss=[]
learning_epoch=1000
embedding_size = 300
loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),1e-3)

for i in range(learning_epoch):
    model.train()
    losss=[]
    acc=0
    total=0
    for num,(score,data) in tqdm(enumerate(train_loader)):
        score=score.cuda()
        data=data.cuda()
        batch_size = score.size(0)
        # print("score",score)
        # print("data",data)
        # score = score.cuda()
        # data = data.cuda()
        predict = model(data)
        # print("predict",predict.shape)
        # print("score,",score.shape)
        loss=loss_function(predict,score)
        # print("loss:{}".format(loss.requires_grad))
        # loss=loss.requires_grad_()
        # print("score:{}".format(score.requires_grad))
        # print("predict:{}".format(predict.requires_grad))
        loss.backward()

        # print(loss.grad)
        losss.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        acc+=(score==torch.argmax(predict,-1)).cpu().sum().item()
        total+=score.size(0)
    print("training epoch %s accuracy is %s loss is %s "%(str(i),str(acc/total),str(np.mean(losss))))
    losss = []
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for num, (score, data) in tqdm(enumerate(test_loader)):
            batch_size=score.size(0)
            embedding_size=300
            score = score.cuda()
            data = data.cuda()
            predict = model(data)
            loss = loss_function(predict,score)
            losss.append(loss.item())
            acc += (score == torch.argmax(predict, -1)).cpu().sum().item()
            total += score.size(0)
        print("test epoch %s accuracy is %s loss is %s " % (str(i), str(acc / total), str(np.mean(losss))))
