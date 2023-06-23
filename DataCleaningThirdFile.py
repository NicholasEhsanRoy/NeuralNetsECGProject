import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz, cheby1
import os
import random


# This function takes in the ecg data(in csv format) from a file(individual patient).
# The ecg data is stripped from its header and then returned as a 2d-numpy array.
def takeData(fileName):
    with open(fileName) as csv_file:
        list1 = []
        list2 = []
        reader = csv.reader(csv_file, delimiter=',')
        flag = True
        for row in reader:
            if flag:
                flag = False
                continue
            list1.append(row[1])
            list2.append(row[2])

    return np.array([list1, list2]).astype(float)


# This function to used to subsample the data from a single patient.
# The first parameter is an array or list and the second parameter is the frequency that the data will be subsampled at.
def subsample(data, Fs):
    list1 = []
    list2 = []
    for i in range(len(data[0])):
        if (i + 1) % Fs == 0:
            list1.append(data[0][i])
            list2.append(data[1][i])
        i += 1
    return np.array([list1, list2])


# This function returns a butterworth bandpass filter object for a given lower and higher frequency cutoff.
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


# This function passes the given data through a butterworth bandpass filter with a given lower and higher frequency cutoff.
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    x = sosfilt(sos, data[0])
    y = sosfilt(sos, data[1])
    return np.array([x, y])


# This function returns a chebychev type 1 bandpass filter object for a given lower and higher frequency cutoff.
def cheby1_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = cheby1(order, 1, [low, high], analog=False, btype='band', output='sos')
    return sos


# This function passes the given data through a chebychev type 1 bandpass filter with a given lower and higher frequency cutoff.
def cheby1_bandpass_filter(data, lowcut, highcut, fs, order=3):
    sos = cheby1_bandpass(lowcut, highcut, fs, order=order)
    x = sosfilt(sos, data[0])
    y = sosfilt(sos, data[1])
    return np.array([x, y])


# This function plots the data from a given patient for a specified range of samples on the x-axis.
# It is possible to specify either the Ml2 reading or the v1/v2/v4/v5 or both to be plotted.
def plot_heart_beats(filtered, start=0, end=3600, ml2_or_v5="both"):
    if (ml2_or_v5 == "ml2"):
        plt.plot(range(start, end), filtered[0][start:end], color="blue")
        plt.legend(['ml2'])
    elif (ml2_or_v5 == "v5"):
        plt.plot(range(start, end), filtered[1][start:end], color="orange")
        plt.legend(['v5'])
    else:
        plt.plot(range(start, end), filtered[0][start:end], color="blue")
        plt.plot(range(start, end), filtered[1][start:end], color="orange")
        plt.legend(['MLII', 'V5'])


# This function normalizes the given data to a range between 0 and 1
def normalise_y_axis(data, start=2100):
    min1 = np.amin(data[0][start:])
    max1 = np.amax(data[0][start:])
    diff1 = max1 - min1

    min2 = np.amin(data[1][start:])
    max2 = np.max(data[1][start:])
    diff2 = max2 - min2

    for i in range(start, len(data[0])):
        data[0][i] = (data[0][i] - min1) / diff1
        data[1][i] = (data[1][i] - min2) / diff2

    return data

#This function creates the training data in the form (x,y) -> (window of length WINDOW_SIZE, annotation).
#It gets the data for one patient file and filters it etc.
#The data is then split into WINDOW_SIZE length segments and all annotations that are in the range of a segement are
#appended as a list making up one example.
#Each of these examples is then checked if they have at least and at most 1 annotations appended. If they have more or
#that specific example is discarded from the dataset.
def getTrainingData(annotationFileName):
    ONE_SECOND = 180

    WINDOW_SIZE = ONE_SECOND * 1

    WINDOW_SIZE = int(WINDOW_SIZE)

    fileLocations = "ecgData/mitbih_database/"

    fileTrainingData = []
    i = 0
    ecgData = takeData(fileLocations + annotationFileName[:3] + ".csv")
    ecgData = butter_bandpass_filter(ecgData, 0.4, 45, 360)
    ecgData = subsample(ecgData, 2)
    ecgData = normalise_y_axis(ecgData)
    ecgData = ecgData[0]
    length = len(ecgData)
    with open(fileLocations + annotationFileName) as annotations:
        annotate = []
        j = 0
        for line in annotations:
            if j == 0:
                j += 1
                continue
            annotate.append([int(line.split()[1]), line.split()[2]])
        for anno in annotate[4:-4]:
            fileTrainingData.append([ecgData[anno[0]//2 - WINDOW_SIZE//2: anno[0]//2 + WINDOW_SIZE//2], anno[1]])


    #     while (i + WINDOW_SIZE) < length:
    #         j = 0
    #         anno = []
    #         for a in annotate[index:]:
    #             if j == 0:
    #                 j += 1
    #                 continue
    #             sampleNum = int(a[0]) // 2
    #             if sampleNum > i:
    #                 if sampleNum < (i + WINDOW_SIZE):
    #                     anno.append(a[1])
    #                     index += 1
    #                 else:
    #                     break
    #
    #         fileTrainingData.append([ecgData[i: i + WINDOW_SIZE], anno])
    #         i += WINDOW_SIZE
    # toRemove = []
    #
    # for i, datapoint in enumerate(fileTrainingData):
    #     if (len(datapoint[1]) != 1):
    #         toRemove.append(i)
    #
    # fileTrainingData = [j for i, j in enumerate(fileTrainingData) if i not in toRemove]

    return fileTrainingData




# path = "./ecgData\mitbih_database"
#
# files = os.listdir(path)
# newFiles = []
# for file in files:
#     if file[-1] == "t":
#         newFiles.append(file)
#
#
# trainingData = []
#
# for x in newFiles:
#     print(x)
#     trainingData = trainingData + getTrainingData(x)
# #
# random.shuffle(trainingData)
#
# trainingSamples = []
# counter = {"N":0,
#            "L":1,
#            "R":2,
#            "A":3,
#            "a":4,
#            "J":5,
#            "S":6,
#            "V":7,
#            "F":8,
#            "[":9,
#            "!":10,
#            "]":11,
#            "e":12,
#            "j":13,
#            "E":14,
#            "/":15,
#            "f":16,
#            "x":17,
#            "Q":18,
#            "|":19}
#
# for i, sample in enumerate(trainingData):
#     if sample[1][0] in counter.keys():
#        sample[1] = counter[sample[1][0]]
# print(counter)


#
# save_training_samples(trainingData)




