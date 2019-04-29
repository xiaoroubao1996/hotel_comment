import os
#读取
def Read(file):
    with open(file,'r') as f:
        newWords = []
        for line in f:
            linestr = line.strip().split(" ")
            newWords.append(linestr)
    return newWords

def Write(newWords, file):
    with open(file, 'w') as f:
        for sentence in newWords:
            for word in sentence:
                f.write(word + " ")
            f.write("\n")

def ReadInt(file):
    with open(file,'r') as f:
        newWords = []
        for line in f:
            lineInt = line.strip().split(" ")
            for index in range(len(lineInt)):
                lineInt[index] = float(lineInt[index])
            newWords.append(lineInt)
    return newWords

def WriteInt(newWords, file):
    with open(file, 'w') as f:
        for sentence in newWords:
            for word in sentence:
                f.write(str(word) + " ")
            f.write("\n")
