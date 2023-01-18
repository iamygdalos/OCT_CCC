import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
def GetImageFiles(dirPath):
    return list(Path(dirPath).rglob('*.[pP][nN][gG]'))

def GetNumberFromFilePath(filePath):
    fileParts = filePath.split('_')
    filePartWithNumber = fileParts[len(fileParts) - 1]
    number = filePartWithNumber.split('.')
    return number[0]

def LoadVGGAndPredict(pathToModel, imgPath, validset):

    dim = (224, 224)

    model = keras.applications.VGG16(input_shape=(224, 224, 3))
    inputs = keras.Input(shape=(224, 224, 3))
    x = model(inputs)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    model.load_weights('Z:\\MA_Inna\\training results\\lulu net\\061221_single_x_view_prepro\\all_20211206_182100\\vgg_ckpt4\\')

    patients = []
    labels = []
    f = open("C:\\Users\\kgr-ik\\Desktop\\MA_Inna\\data\\list_krlm_TRAIN.txt", "r")
    for l in f:
        splits = l.split(';')
        patients.append(splits[0])
        labels.append(splits[1])
    f.close()

    validPredictionsVal = []
    validPredictionsValCscan = []
    validPredictionsValBscan = []
    validAllTrueTrainLabels = []
    validAllTrueTrainLabelsCscan = []
    validAllTrueTrainLabelsBscan = []
    for val in validset:
        pat = patients[val]
        lab = labels[val]

        patPath = imgPath

        if lab == "T":
            patPath = patPath + "\\abnormal\\"
            validAllTrueTrainLabelsCscan.append(1)
        else:
            patPath = patPath + "\\normal\\"
            validAllTrueTrainLabelsCscan.append(0)

        patPath = patPath + pat

        validPatientFiles = GetImageFiles(patPath)
        validPatientPredictForCScan = []
        bScanPredict = []
        lenValidPatients = len(validPatientFiles)
        fileNum = 0
        for file in validPatientFiles:
            fileNum = fileNum + 1
            number = GetNumberFromFilePath(str(file))

            if (number == '0' and len(bScanPredict) != 0):
                tmpBScan = np.mean(bScanPredict)
                validPredictionsValBscan.append(tmpBScan)

                if lab == "T":
                    validAllTrueTrainLabelsBscan.append(1)
                else:
                    validAllTrueTrainLabelsBscan.append(0)

                bScanPredict = []

            img = cv2.imread(str(file), cv2.IMREAD_COLOR)
            img = img/255
            img = cv2.resize(img, dim)
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)

            validPredictionsVal.append(pred)

            validPatientPredictForCScan.append(pred)
            bScanPredict.append(pred)

            if lab == "T":
                validAllTrueTrainLabels.append(1)
            else:
                validAllTrueTrainLabels.append(0)

            if (fileNum == lenValidPatients and len(bScanPredict) != 0):
                tmpBScan = np.mean(bScanPredict)
                validPredictionsValBscan.append(tmpBScan)

                if lab == "T":
                    validAllTrueTrainLabelsBscan.append(1)
                else:
                    validAllTrueTrainLabelsBscan.append(0)

                bScanPredict = []

        validFinalPatientPredict = np.mean(validPatientPredictForCScan)
        validPredictionsValCscan.append(validFinalPatientPredict)

    testPatients = []
    testLabels = []
    f = open("C:\\Users\\kgr-ik\Desktop\\MA_Inna\\data\\list_krlm_TEST.txt", "r")
    for l in f:
        splits = l.split(';')
        testPatients.append(splits[0])
        testLabels.append(splits[1])
    f.close()

    i = 0
    testPredictionsVal = []
    testPredictionsValBscan = []
    testPredictionsValCscan = []
    testAllTrueTrainLabels = []
    testAllTrueTrainLabelsBscan = []
    testAllTrueTrainLabelsCscan = []
    for pat in testPatients:
        lab = testLabels[i]
        i = i + 1
        patPath = imgPath

        if lab == "T":
            patPath = patPath + "\\abnormal\\"
            testAllTrueTrainLabelsCscan.append(1)
        else:
            patPath = patPath + "\\normal\\"
            testAllTrueTrainLabelsCscan.append(0)

        patPath = patPath + pat

        testPatientFiles = GetImageFiles(patPath)

        testPatientPredict = []
        bScanPredict = []
        lenTestPatients = len(testPatientFiles)
        fileNum = 0

        for file in testPatientFiles:
            fileNum = fileNum + 1
            number = GetNumberFromFilePath(str(file))

            if (number == '0' and len(bScanPredict) != 0):
                tmpBScan = np.mean(bScanPredict)

                testPredictionsValBscan.append(tmpBScan)

                if lab == "T":
                    testAllTrueTrainLabelsBscan.append(1)
                else:
                    testAllTrueTrainLabelsBscan.append(0)


                bScanPredict = []

            img = cv2.imread(str(file), cv2.IMREAD_COLOR)
            img = img/255
            img = cv2.resize(img, dim)
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)
            testPredictionsVal.append(pred)

            testPatientPredict.append(pred)
            bScanPredict.append(pred)

            if lab == "T":
                testAllTrueTrainLabels.append(1)
            else:
                testAllTrueTrainLabels.append(0)

            if (fileNum == lenTestPatients and len(bScanPredict) != 0):
                tmpBScan = np.mean(bScanPredict)
                testPredictionsValBscan.append(tmpBScan)

                if lab == "T":
                    testAllTrueTrainLabelsBscan.append(1)
                else:
                    testAllTrueTrainLabelsBscan.append(0)

                bScanPredict = []

        finalPatientPredict = np.mean(testPatientPredict)
        testPredictionsValCscan.append(finalPatientPredict)


    keras.backend.clear_session()
    return validAllTrueTrainLabels, validPredictionsVal, validAllTrueTrainLabelsBscan, validPredictionsValBscan, validAllTrueTrainLabelsCscan, validPredictionsValCscan, testAllTrueTrainLabels, testPredictionsVal, testAllTrueTrainLabelsBscan, testPredictionsValBscan, testAllTrueTrainLabelsCscan, testPredictionsValCscan