import torch
import numpy as np
from torch.autograd import Variable


class LSTM_Unit(torch.nn.Module):
    def __init__(self, embedding_size):
        super(LSTM_Unit, self).__init__()
        self.weightf = torch.nn.Linear(2 * embedding_size, embedding_size)  # forget gate
        self.weighti = torch.nn.Linear(2 * embedding_size, embedding_size)  # memory gate
        self.weightc = torch.nn.Linear(2 * embedding_size, embedding_size)  # cell status
        self.weighto = torch.nn.Linear(2 * embedding_size, embedding_size)  # out status
        self.sigmoid = torch.nn.Sigmoid()
        self.cell_status=Variable(torch.zeros((1, embedding_size)), requires_grad=True).cuda()
        self.tanh = torch.nn.Tanh()

    def forward(self, state: torch.Tensor, input: torch.Tensor, cons: torch.Tensor):
        """

        :param state: last state
        :param input: word embedding
        :param cons: reference
        """
        # print("state",state.shape)
        # print("input",input.shape)

        concat_status = torch.cat((state, input), -1)
        forget_status = self.weightf(concat_status)
        forget_status = self.sigmoid(forget_status)
        mem_gate = self.sigmoid(self.weighti(concat_status))
        tmp_cell_status = self.tanh(self.weightc(concat_status))
        cell_gate = mem_gate.mul(tmp_cell_status)
        out_cell_status = cons.mul(forget_status) + cell_gate
        out_hidden = self.tanh(out_cell_status).mul(forget_status)
        return out_hidden, out_cell_status


class LSTM(torch.nn.Module):
    def __init__(self, embeddings_size, layer_num,sentence_length, label_size,dropoutrate):
        super(LSTM, self).__init__()
        self.forward_model = torch.nn.ModuleList([LSTM_Unit(embeddings_size) for _ in range(layer_num)])
        self.back_model = torch.nn.ModuleList([LSTM_Unit(embeddings_size) for _ in range(layer_num)])
        # self.linear=torch.nn.Linear(2*embeddings_size,label_size)
        self.word_linear = torch.nn.Linear(2 * embeddings_size, 1)
        self.predict_linear = torch.nn.Linear(sentence_length, label_size)
        self.layer_num=layer_num
        self.embedding_size=embeddings_size
        self.fdropout=[torch.nn.Dropout(dropoutrate) for i in range(layer_num)]
        self.bdropout=[torch.nn.Dropout(dropoutrate) for i in range(layer_num)]
        self.fcell=[torch.nn.Dropout(dropoutrate) for i in range(layer_num)]
        self.bcell=[torch.nn.Dropout(dropoutrate) for i in range(layer_num)]
        self.dropout=torch.nn.Dropout(dropoutrate)
    def forward(self, inputs: torch.Tensor, cell_states, hidden_states, back_states, back_hidden):
        # print(input.shape)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        # embedding_size = inputs.size(2)
        inputs = inputs.permute([1, 0, 2])
        output = []
        # print("Cell_states:{}".format(cell_states.requires_grad))
        # print("hidden_states:{}".format(hidden_states.requires_grad))
        # print("back_states:{}".format(back_states.requires_grad))
        # print("back_hidden:{}".format(back_hidden.requires_grad))
        forward=[]
        backward=[]
        tmp_fce=[cell_states]*self.layer_num
        tmp_bce=[cell_states]*self.layer_num
        fhidden=[hidden_states]*self.layer_num
        bhidden=[hidden_states]*self.layer_num
        # print("hidden",hidden_states.shape)
        # print("cell",cell_states.shape)
        # print("inputs",inputs.shape)
        for word in range(seq_len):
            forward_in = inputs[word]
            backward_inputs = inputs[seq_len - word - 1]
            for i in range(self.layer_num):
                fhidden[i], tmp_fce[i] = self.forward_model[i](fhidden[i], forward_in, tmp_fce[i])
                forward_in=self.fdropout[i](fhidden[i])
                bhidden[i], tmp_bce[i] = self.back_model[i](bhidden[i], backward_inputs, tmp_bce[i])
                backward_inputs=self.bdropout[i](bhidden[i])

            forward.append(forward_in)
            backward.append(backward_inputs)
        forward=torch.stack(forward,dim=0)
        backward.reverse()
        backward=torch.stack(backward,dim=0)
        out_status = torch.cat((forward,backward), -1)
            # print("Cell_states{}:{}".format(i,cell_states.requires_grad))
            # print("hidden_states{}:{}".format(i,hidden_states.requires_grad))
            # print("back_states{}:{}".format(i,back_states.requires_grad))
            # print("back_hidden{}:{}".format(i,back_hidden.requires_grad))
        # print("output_state1:{}".format(output.requires_grad))
        output = out_status.permute(1, 0, 2)
        # print("output_state2:{}".format(output.requires_grad))
        output = self.word_linear(output)
        output=self.dropout(output)
        # print("output_state3:{}".format(output.requires_grad))
        output = self.predict_linear(output.view(batch_size, -1))
        # print("output_state4:{}".format(output.requires_grad))
        return output
