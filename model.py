import torch
import numpy
from Bi_LSTM import LSTM
class Modle(torch.nn.Module):
    def __init__(self, seq_len, label_size):
        super(Modle,self).__init__()
        weight=[]
        with open("./data/glove.840B.300d.txt") as f:
            line=f.readline()
            while line:
                data=line.strip().split(' ')[1:]
                sen=[float(i) for i in data]
                weight.append(sen)

        weights=numpy.array(weight)
        embedding_size=300
        vocab_size=len(weights)
        self.model=LSTM(embedding_size,seq_len,label_size)
        self.embedding=torch.nn.Embedding(vocab_size,embedding_size)
        self.embedding.from_pretrained(torch.from_numpy(weights))
    def forward(self,inputs):
        inputs=self.embedding(inputs)
        outs=self.model(inputs)
        return outs