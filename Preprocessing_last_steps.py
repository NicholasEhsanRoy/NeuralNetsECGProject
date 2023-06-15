import DataCleaningThirdFile as dct
import os
import numpy as np

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

    reference = {"N":0,
               "L":1,
               "R":2,
               "A":3,
               "a":4,
               "J":5,
               "S":6,
               "V":7,
               "F":8,
               "[":9,
               "!":10,
               "]":11,
               "e":12,
               "j":13,
               "E":14,
               "/":15,
               "f":16,
               "x":17,
               "Q":18,
               "|":19,
               "~":20,
               "+":21,
               "\"":22,}

    for sample in trainingData:
        if sample[1][0] in reference.keys():
           sample[1] = reference[sample[1][0]]
        else:
            print("didnt work with annotation: " + str(sample[1][0]))
    return trainingData


trainingData = process_data()
trainingData = np.array(trainingData, dtype=object)

np.savetxt('training_examples.txt', trainingData)