import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy
import torchsummary
from Bi_LSTM import LSTM


class Modle(torch.nn.Module):
    def __init__(self, seq_len, label_size,layer_num,target_size,dropout_rate):
        super(Modle, self).__init__()
        weight = []
        print("reading glove")
        with open("./data/SST-cla/glove.txt") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if line=='\n':
                    continue
                data = line.strip().split(' ')[1:]
                sen = [float(i) for i in data]
                weight.append(sen)
                assert len(sen)==300
        del lines
        self.target_size=target_size
        self.convert=torch.nn.Linear(300,target_size)
        # self.dropout=torch.nn.Dropout(dropout_rate)
        weights = numpy.array(weight)
        vocab_size = weights.shape[0]
        print("vocab_size", vocab_size)
        self.model = LSTM(target_size, layer_num,seq_len, label_size,dropout_rate)
        self.embedding = torch.nn.Embedding(vocab_size, 300)
        # self.embedding.weight.requires_grad
        self.cell_states = Variable(torch.zeros((1, target_size)), requires_grad=False).cuda()
        self.hidden_states =Variable(torch.zeros((1, target_size)), requires_grad=False).cuda()
        self.back_states = Variable(torch.zeros((1, target_size)), requires_grad=False).cuda()
        self.back_hidden =Variable(torch.zeros((1, target_size)), requires_grad=False).cuda()
        self.embedding.from_pretrained(torch.from_numpy(weights))
        # self.LSTM = LSTM(target_size,layer_num,seq_len,label_size,dropout_rate)
        self.sig=torch.nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        inputs = inputs.cuda()
        inputs=self.convert(inputs)
        # print("inputs:{}".format(inputs.requires_grad))
        outs = self.model(inputs, self.cell_states.repeat(inputs.size(0), 1),
                          self.hidden_states.repeat(inputs.size(0), 1), self.back_states.repeat(inputs.size(0), 1),
                          self.back_hidden.repeat(inputs.size(0), 1))
        outs=self.sig(outs)
        return outs
