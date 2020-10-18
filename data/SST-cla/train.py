import torch
from tqdm import tqdm
from torch.autograd import  Variable
from T_f import Modle
from reader import Reader
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("create train reader")
train_reader=Reader("./data/SST-cla/train.txt",56)
test_reader=Reader("./data/SST2/test.txt",56)
print("create test reader")
dev_reader=Reader("./data/SST2/dev.txt",56)
print("max sentence length",train_reader.max_line)
print(test_reader.max_line)
train_loader=torch.utils.data.DataLoader(train_reader,shuffle=True,batch_size=25)
test_loader=torch.utils.data.DataLoader(test_reader,shuffle=True,batch_size=25)
dev_loader=torch.utils.data.DataLoader(dev_reader,shuffle=True,batch_size=25)
print("building model")
model=Modle(train_reader.max_line,5,2,120,0.1)
print("model finished")
model=model.cuda()
model.train()
loss=[]
model.zero_grad()
learning_epoch=20
embedding_size = 300
loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),0.01)
tr=open("train.txt",'w')
file = open("tests.txt", 'w')
for i in range(learning_epoch):
    model.train()
    losss=[]
    acc=0
    total=0
    for num,(score,data) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        model.train()
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
        optimizer.step()
        # for each in zip(score,predict):
        #     tr.write(str(each)+'\n')
        # print(loss.grad)
        losss.append(loss.item())
        # tmp=zip(score,torch.argmax(predict,-1))
        # print(tmp)
        acc+=(score==torch.argmax(predict,-1)).cpu().sum().item()
        total+=score.size(0)

    print("training epoch %s accuracy is %s loss is %s " % (str(i), str(acc / total), str(np.mean(losss))))
    # if num %100==0:
    torch.save(model.state_dict(), "model.bin")
    losss2 = []
    acc2 = 0
    total2 = 0
    model.eval()
    with torch.no_grad():
        for num, (score, data) in tqdm(enumerate(test_loader)):
            batch_size=score.size(0)
            embedding_size=300
            score = score.cuda()
            data = data.cuda()
            predict = model(data)
            # print(predict.shape)
            loss = loss_function(predict,score)
            losss2.append(loss.item())
            acc2 += (score == torch.argmax(predict, -1)).cpu().sum().item()
            total2 += score.size(0)
        print("test epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))
        losss2 = []
        acc2 = 0
        total2 = 0
        for num, (score, data) in tqdm(enumerate(dev_loader)):
            batch_size=score.size(0)
            embedding_size=300
            score = score.cuda()
            data = data.cuda()
            predict = model(data)
            loss = loss_function(predict,score)
            losss2.append(loss.item())
            acc2 += (score == torch.argmax(predict, -1)).cpu().sum().item()
            total2 += score.size(0)
            # for each in zip(score,predict,data):
            #     file.write(str(each)+"\n")
        print("dev epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))
