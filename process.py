from tqdm import tqdm
dict={}
with open("./data/glove.840B.300d.txt",'r',encoding='utf-8') as f:
    lines=f.readlines()
    for  line in tqdm(lines):
        word=line.split(' ')[0]
        dict[word]=len(dict.keys())

file_path="./data/SST/dev.txt"
with open(r"data/SST-cla/dev.txt",'w') as out:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        score = []
        sentence = []
        for line in tqdm(lines):
            score = int(line[1])
            sentence = []
            char_next = False
            space = True
            for char in line.strip():
                if char_next:
                    char_next = False
                    continue
                if char == '(':  # 左等于，先判断这里
                    char_next = True
                    space = True
                elif char == ')':
                    space = True
                    continue
                elif char == ' ':
                    space = True
                elif space == True:
                    if sentence == [] or sentence[-1] != ['']:
                        sentence.append('')
                    sentence[-1] += char
                    space = False
                else:
                    sentence[-1] += char
            te=[]
            for word in sentence:

                if word in dict.keys():
                    te.append(str(dict[word]))
                else:
                    print("word %s missed"%word)
                    te.append(str(dict["<unk>"]))
            out.write("%s %s\n"%(str(score),' '.join(te)))