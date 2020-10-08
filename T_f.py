import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy
import torchsummary
from torch.nn import LSTM


class Modle(torch.nn.Module):
    def __init__(self, seq_len, label_size,layer_num,target_size,dropout):
        super(Modle, self).__init__()
        weight = []
        print("reading glove")
        with open("./data/SST-cla/glove.txt") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                data = line.strip().split(' ')[1:]
                if data == []:
                    continue
                sen = [float(i) for i in data]
                weight.append(sen)
                assert len(sen)==300
        del lines
        weights = numpy.array(weight)
        embedding_size = 300
        vocab_size = weights.shape[0]
        print("vocab_size", vocab_size)
        self.model = LSTM(embedding_size,target_size, layer_num,dropout=dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.embedding.from_pretrained(torch.from_numpy(weights))
        self.linear1=torch.nn.Linear(target_size,1)
        self.linear2=torch.nn.Linear(seq_len,label_size)

    def forward(self, inputs):
        batch_size=inputs.size(0)
        inputs = self.embedding(inputs)
        inputs=inputs.permute((1,0,2))
        inputs = inputs.cuda()
        # print("inputs:{}".format(inputs.requires_grad))
        outs,(_,_) = self.model(inputs)
        outs=outs.permute((1,0,2))
        outs=self.linear1(outs)
        outs=self.linear2(outs.view(batch_size,-1))


        return outs
