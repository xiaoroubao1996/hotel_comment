import os
#读取
def Read():
    with open('./corpus/corpus.txt','r') as f:
        newWords = []
        for line in f:
            linestr = line.strip().split(" ")
            newWords.append(linestr)
    return newWords

def Write(newWords):
    with open('./corpus/corpus.txt', 'w') as f:
        for sentence in newWords:
            for word in sentence:
                f.write(word + " ")
            f.write("\n")

