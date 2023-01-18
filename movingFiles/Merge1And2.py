import os
import shutil
import re

# CreateDatasetKRLM() takes two datasets and merges them
# dataset folder has to be in: basefolder/diagnose/patientcode/imgNum
# imgNum is orientation from which data was taken out of C-Scan

imgNum = 2

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def CreateDatasetKRLM(baseFolder1, baseFolder2, outputFolder):
    if not (os.path.exists(outputFolder)):
        os.mkdir(outputFolder)

    for diagnoseDir in os.listdir(baseFolder1):
        if not (os.path.exists(outputFolder + '\\' + diagnoseDir)):
            os.mkdir(outputFolder + '\\' + diagnoseDir)

        for patient in os.listdir(baseFolder1 + '\\' + diagnoseDir):
            print(patient)

            if not (os.path.exists(outputFolder + '\\' + diagnoseDir + '\\' + patient)):
                os.mkdir(outputFolder + '\\' + diagnoseDir + '\\' + patient)

            for imgNum in os.listdir(baseFolder1 + '\\' + diagnoseDir + '\\' + patient + '\\'):
                for fileName in os.listdir(baseFolder1 + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\'):
                    shutil.copy(
                        baseFolder1 + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                        outputFolder + '\\' + diagnoseDir + '\\' + patient+ '\\' + fileName)

    for diagnoseDir in os.listdir(baseFolder2):

        if not (os.path.exists(outputFolder + '\\' + diagnoseDir)):
            os.mkdir(outputFolder + '\\' + diagnoseDir)

        for patient in os.listdir(baseFolder2 + '\\' + diagnoseDir):
            print(patient)

            if not (os.path.exists(outputFolder + '\\' + diagnoseDir + '\\' + patient)):
                os.mkdir(outputFolder + '\\' + diagnoseDir + '\\' + patient)

            for imgNum in os.listdir(baseFolder2 + '\\' + diagnoseDir + '\\' + patient + '\\'):
                for fileName in os.listdir(
                        baseFolder2 + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\'):
                    shutil.copy(
                        baseFolder2 + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                        outputFolder + '\\' + diagnoseDir + '\\' + patient + '\\' + fileName)


# Folder to exported OCT
outputFolder = 'D:\\MA_Luisa\\data\\KRLMvsNormal_1And2_Singles_MaskedSquared\\'
# Folder to save Dataset
baseFolder1 = 'D:\\MA_Luisa\\data\\exportKRLM_1_Singles_MaskedSquared\\'
baseFolder2 = 'D:\\MA_Luisa\\data\\exportKRLM_2_Singles_MaskedSquared\\'

CreateDatasetKRLM(baseFolder1, baseFolder2, outputFolder)









