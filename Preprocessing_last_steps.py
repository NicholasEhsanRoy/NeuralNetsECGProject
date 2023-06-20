import DataCleaningThirdFile as dct
import os
import numpy as np
import random

def process_data():
    trainingData = []
    path = "./ecgData\mitbih_database"

    files = os.listdir(path)
    newFiles = []
    for file in files:
        if file[-1] == "t":
            newFiles.append(file)



    for x in newFiles:
        print(x)
        trainingData = trainingData + dct.getTrainingData(x)

    return trainingData

def one_hot_encode(trainingData):
    reference = {"N": 0,
                 "L": 1,
                 "R": 2,
                 "V": 3,
                 "/": 4,
                 "J": 5,
                 "S": 6,
                 "A": 7,
                 "F": 8,
                 "[": 9,
                 "!": 10,
                 "]": 11,
                 "e": 12,
                 "j": 13,
                 "E": 14,
                 "a": 15,
                 "f": 16,
                 "x": 17,
                 "Q": 18,
                 "|": 19,
                 "~": 20,
                 "+": 21,
                 "\"": 22, }

    for sample in trainingData:
        if sample[1][0] in reference.keys():
            vector = [0, 0, 0, 0, 0]
            vector[reference[sample[1][0]]] = 1
            sample[1] = vector
        else:
            print("didnt work with annotation: " + str(sample[1][0]))
    return trainingData

def sample_targets(trainingData):
    new_trainingData = []
    random.shuffle(trainingData)
    counter = {'V': 0,
               'R': 0,
               'L': 0,
               'N': 0,
               '/': 0}
    for sample in trainingData:
        if sample[1][0] in counter.keys() and counter[sample[1][0]] < 2000:
            new_trainingData.append(sample)
        if (counter['V'] + counter['R'] + counter['L'] + counter['N'] + counter['/']) >= 1000:
            return new_trainingData
    return new_trainingData


trainingData = process_data()
trainingData = sample_targets(trainingData)
trainingData = one_hot_encode(trainingData)

print(len(trainingData))
print(len(trainingData[1][1]))
print((trainingData[1][1]))