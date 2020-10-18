import torch
import numpy as np
from torch.autograd import Variable


class LSTM_Unit(torch.nn.Module):
    def __init__(self,hidden_size):
        super(LSTM_Unit, self).__init__()
        self.linear=torch.nn.Linear(hidden_size,4*hidden_size)
        self.linear_hidden=torch.nn.Linear(hidden_size,4*hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        # self.cell_status=Variable(torch.zeros((1, embedding_size)), requires_grad=True).cuda()
        self.tanh = torch.nn.Tanh()
        print("hidden_size")

    def forward(self, state: torch.Tensor, input: torch.Tensor, cons: torch.Tensor):
        """

        :param state: last state
        :param input: word embedding
        :param cons: reference
        """
        # print("state",state.shape)
        # print("input",input.shape)
        # print(input.shape)
        # print(state.shape)
        feature=self.linear(input)+self.linear_hidden(state)
        feature.squeeze()
        ingate,forget_gate,cell_gate,cellstat=feature.chunk(4,1)
        ingate=self.sigmoid(ingate)
        forget_gate=self.sigmoid(forget_gate)
        cell_gate=self.tanh(cell_gate)
        cellstat=self.sigmoid(cellstat)
        cell=torch.mul(cons,forget_gate)+torch.mul(ingate,cell_gate)
        out=torch.mul(self.tanh(cell),cellstat)
        return out,cell


class LSTM(torch.nn.Module):
    def __init__(self, embeddings_size, layer_num,embe_size, label_size,dropoutrate):
        super(LSTM, self).__init__()
        self.layer_num=layer_num
        self.model=torch.nn.ModuleList([LSTM_Unit(embe_size) for i in range(layer_num)])
        self.predict_linear=torch.nn.Linear(embe_size,label_size)
        self.hidden_size=embe_size
        self.dropout=torch.nn.Dropout(dropoutrate)
    def forward(self, inputs: torch.Tensor):
        # print(input.shape)
        self.begin_state= Variable(torch.zeros(self.layer_num,inputs.size(0), self.hidden_size).cuda())
        self.hidden_state=Variable(torch.zeros(self.layer_num,inputs.size(0), self.hidden_size).cuda())
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        # embedding_size = inputs.size(2)
        inputs = inputs.permute([1, 0, 2])
        next_inputs=inputs
        for j in range(self.layer_num):
            hidden=self.hidden_state[j]
            state=self.begin_state[j]
            current_inputs=next_inputs
            next_inputs=[]
            for i in range(seq_len):
                # if j!=0:
                    # hidden=self.dropout(hidden)
                hidden,state=self.model[j](hidden,current_inputs[i],state)
                if j!=self.layer_num-1:
                    hidden=self.dropout(hidden)
                next_inputs.append(hidden)
        output=torch.stack(next_inputs,0)
        output=output.permute([1,2,0])
        output=torch.max_pool1d(output,output.size(2)).squeeze(2)
        output = self.predict_linear(output)
        # print("output_state4:{}".format(output.requires_grad))
        return output
