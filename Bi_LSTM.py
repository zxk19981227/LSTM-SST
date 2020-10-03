import torch


class LSTM_Unit(torch.nn.Module):
    def __init__(self, embedding_size):
        super(LSTM_Unit, self).__init__()
        self.weightf = torch.nn.Linear(2 * embedding_size, embedding_size)  # forget gate
        self.weighti = torch.nn.Linear(2 * embedding_size, embedding_size)  # memory gate
        self.weightc = torch.nn.Linear(2 * embedding_size, embedding_size)  # cell status
        self.weighto = torch.nn.Linear(2 * embedding_size, embedding_size)  # out status
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, state: torch.Tensor, input: torch.Tensor, cons:torch.Tensor):
        """

        :param state: last state
        :param input: word embedding
        :param cons: reference
        """
        concat_status = torch.cat((state, input), -1)
        forget_status = self.weightf(concat_status)
        forget_status = self.sigmoid(forget_status)
        mem_gate = self.sigmoid(self.weighti(concat_status))
        tmp_cell_status = self.tanh(self.weightc(concat_status))
        cell_gate = mem_gate.mul(tmp_cell_status)
        out_cell_status=cons.mul(forget_status)+cell_gate
        out_hidden=self.tanh(out_cell_status).mul(forget_status)
        return out_hidden,out_cell_status


class LSTM(torch.nn.Module):
    def __init__(self,embeddings_size,sentence_length,label_size):
        super(LSTM,self).__init__()
        self.forward_model=torch.nn.ModuleList([LSTM_Unit(embeddings_size) for _ in range(sentence_length)])
        self.back_model=torch.nn.ModuleList([LSTM_Unit(embeddings_size) for _ in range(sentence_length)])
        self.linear=torch.nn.Linear(2*embeddings_size,label_size)
        self.word_linear=torch.nn.Linear(embeddings_size,1)
        self.predict_linear=torch.nn.Linear(sentence_length,label_size)
    def forward(self, inputs:torch.Tensor):
        batch_size=inputs.size(0)
        seq_len=inputs.size(1)
        embedding_size=inputs.size(2)
        cell_states=torch.zeros((batch_size,embedding_size))
        hidden_states=torch.zeros((batch_size,embedding_size))
        back_states=torch.zeros((batch_size,embedding_size))
        back_hidden=torch.zeros((batch_size,embedding_size))
        inputs=inputs.permute([1,0,2])
        output=[]
        for i in range(seq_len):
            hidden_states,cell_status=self.forward_model[i](hidden_states,inputs[i],cell_states)
            back_hidden,back_states=self.back_model[i](back_hidden,inputs[seq_len-i-1],back_states)
            out_status=torch.cat((hidden_states,back_hidden),-1)
            output.append(out_status)
        output=torch.tensor(output).cuda()
        output=self.word_linear(output)
        output=self.predict_linear(output.view(batch_size,output.shape[1]))
        return torch.tensor(output)
