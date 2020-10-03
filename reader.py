import torch
class Reader(torch.utils.data.Dataset):
    def __init__(self,file_path,max_len):
        with open(file_path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            self.score=[]
            self.sentence=[]
            max_line=0
            for line in lines:
                self.score=int(line.split(' ')[0])
                words=line.split(' ')[1:]
                words=[int(i) for i in words]
                if len(words)>max_line:
                    max_line=len(words)
                self.sentence.append(words)
            max_line=max(max_len,max_line)
            for i in range(len(self.sentence)):
                while len(self.sentence[i])<max_line:
                    self.sentence[i].append(0)
        self.max_line=max(max_line,max_len)
        assert len(self.score)==len(self.sentence)
        self.sentence=torch.tensor(self.score)
        self.score=torch.tensor(self.score)
    def __getitem__(self, item):
        return self.score[item],self.sentence[item]




