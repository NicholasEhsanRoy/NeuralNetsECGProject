import math

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
            vector = np.zeros(5)
            vector[reference[sample[1][0]]] = 1
            sample[1] = vector
        else:
            print("didnt work with annotation: " + str(sample[1][0]))
    return trainingData

def sample_targets(trainingData):
    new_trainingData = []
    random.shuffle(trainingData)
    indices = {'V': 0,
               'R': 1,
               'L': 2,
               'N': 3,
               '/': 4}
    new_data = [[],[],[],[],[]]
    for sample in trainingData:
        if sample[1][0] in indices.keys():
            new_data[indices[sample[1][0]]].append(sample)
    lowest = math.inf
    for annotations in new_data:
        if len(annotations) < lowest:
            lowest = len(annotations)
    for annotations in new_data:
        new_trainingData.extend(annotations[:lowest])
    random.shuffle(new_trainingData)
    return new_trainingData


def convert_to_nparray(trainingData):
    samples = np.empty(shape=(len(trainingData),180))
    targets = np.empty(shape=(len(trainingData),5))

    for i, training in enumerate(trainingData):
        samples[i] = training[0]
        targets[i] = training[1]
    np.save("targets_centered_max_samples", targets)
    np.save("samples_centered_max_samples", samples)

def get_usable_trainingdata():
    trainingData = process_data()
    trainingData = sample_targets(trainingData)
    trainingData = one_hot_encode(trainingData)
    convert_to_nparray(trainingData)

get_usable_trainingdata()

