import os
import shutil
import re

# CreateDataset creates Copy of dataset with lower amount of images
# imgNum is orientation from which data was taken out of C-Scan
# third parameter of CreateDataset 'num' describes: every 'num' image from whole dataset. For example num=2 takes every second image. num = -1 takes all images

imgNum = 1

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def CopyAll(dataFolder, outputFolder):
    for fileName in os.listdir(dataFolder):
        shutil.copy(dataFolder + '\\' + fileName, outputFolder)

def CopySingles(dataFolder, outputFolder, num):
    fileNames = os.listdir(dataFolder)
    fileNames.sort(key=natural_keys)
    for i in range(int(len(fileNames) / num)):
        fileName = fileNames[i*num]
        shutil.copy(dataFolder + '\\' + fileName, outputFolder)

    # fileName = fileNames[200]
    # shutil.copy(dataFolder + '\\' + fileName, outputFolder)

    #fileName = fileNames[500]
    #shutil.copy(dataFolder + '\\' + fileName, outputFolder)

def CopyForDiagnosisAndSettype(patientcode, base_folder, strDiag, outputFolder, num):
    data_folder = base_folder + '\\' + strDiag + '\\' + patientcode + '\\' + str(imgNum) + '\\'

    outputFolderTemp = outputFolder + '\\' + strDiag
    if not (os.path.exists(outputFolderTemp)):
        os.mkdir(outputFolderTemp)

    outputFolderTemp = outputFolder + '\\' + strDiag + '\\' + patientcode
    if not (os.path.exists(outputFolderTemp)):
        os.mkdir(outputFolderTemp)

    outputFolderTemp = outputFolder + '\\' + strDiag + '\\' + patientcode + '\\' + str(imgNum) + '\\'
    if not (os.path.exists(outputFolderTemp)):
        os.mkdir(outputFolderTemp)

    if num == 0:
        CopyAll(data_folder, outputFolderTemp)
    else:
        CopySingles(data_folder, outputFolderTemp, num)


def CreateDataset(base_folder, outputFolder, num):
    if not (os.path.exists(outputFolder)):
        os.mkdir(outputFolder)

    if not (os.path.exists(outputFolder + '\\abnormal')):
        os.mkdir(outputFolder + '\\abnormal')

    if not (os.path.exists(outputFolder + '\\normal')):
        os.mkdir(outputFolder + '\\normal')

    all_normal = os.listdir(base_folder + '\\normal')
    all_abnormal = os.listdir(base_folder + '\\abnormal')

    for patient in all_normal:
        if 'temp' in patient:
            continue

        CopyForDiagnosisAndSettype(patient, base_folder, 'normal', outputFolder, num)

    for patient in all_abnormal:
        if 'temp' in patient:
            continue

        CopyForDiagnosisAndSettype(patient, base_folder, 'abnormal', outputFolder, num)

# Folder to exported OCT
base_folder = 'D:\\MA_Luisa\\data\\exportKRLM_1\\'
# Folder to save Dataset
outputFolder = 'D:\\MA_Luisa\\data\\exportKRLM_1_Single\\'

#CreateSetAll(base_folder, outputFolder)
CreateDataset(base_folder, outputFolder, 500)









