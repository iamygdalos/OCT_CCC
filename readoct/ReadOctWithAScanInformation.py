import zipfile
import os
import cv2
import numpy as np
import time
import xml.etree.ElementTree as ET
import shutil
import pathlib as pat


#ReadOcts() reads .oct file from folder, scales the intensity based on A-Scan information (high contrast) and saves them to output folder


def UnzipOctFile(folder, filename, folder_export):
    tempDir = folder_export + 'temp'
    if (os.path.exists(tempDir)):
        return tempDir

    os.mkdir(tempDir)

    with zipfile.ZipFile(folder + '\\' + filename, 'r') as zip_ref:
        zip_ref.extractall(tempDir)

    return tempDir

def GetImageSizes(filepath):
    xmlPath = filepath + '\\Header.xml'

    tree = ET.parse(xmlPath)
    root = tree.getroot()

    imageSizes = [int(root[0][2].attrib['SizeZ']), int(root[0][2].attrib['SizeX']),
                          int(root[0][2].attrib['SizeY'])]


    k = root[1][4][0].text

    ranges = [float(root[1][4][0].text), float(root[1][4][1].text),
                          float(root[1][4][2].text)]

    return imageSizes, ranges

def ReadRealDataAndSaveImage(filename, filepath, fileExport):
    imageSizes, ranges = GetImageSizes(filepath)

    dataPath = filepath + '\\data\\Intensity.data'

    with open(dataPath, 'r') as f:
        raw = np.fromfile(f, np.float32, imageSizes[0]*imageSizes[1]*imageSizes[2])

    data = raw.reshape((imageSizes[0], imageSizes[1], imageSizes[2]), order='F')
    data = data.astype(int)

    max_data = np.max(data[:, :, 0], 0)
    meanMax1 = np.mean(max_data)

    max_data = np.max(data[:, :, int(imageSizes[2] / 2)], 0)
    meanMax2 = np.mean(max_data)

    max_data = np.max(data[:, :, imageSizes[2]-1], 0)
    meanMax3 = np.mean(max_data)

    meanMax = np.mean([meanMax1, meanMax2, meanMax3])

    min_data = np.mean(data[:, :, 0], 0)
    meanMin1 = np.mean(min_data)

    min_data = np.mean(data[:, :, int(imageSizes[2] / 2)], 0)
    meanMin2 = np.mean(min_data)

    min_data = np.mean(data[:, :, imageSizes[2] - 1], 0)
    meanMin3 = np.mean(min_data)

    meanMin = np.min([meanMin1, meanMin2, meanMin3])

    max_data = meanMax
    min_data = meanMin

    data[data > max_data] = max_data
    data[data < min_data] = min_data

    img = (255.0 / (max_data - min_data)) * (data - min_data)
    print(max_data)

    if not (os.path.exists(fileExport + '\\0\\')):
        print('creating 0')
        os.mkdir(fileExport + '\\0\\')


    if not (os.path.exists(fileExport + '\\1\\')):
        print('creating 1')
        os.mkdir(fileExport + '\\1\\')

    if not (os.path.exists(fileExport + '\\2\\')):
        print('creating 2')
        os.mkdir(fileExport + '\\2\\')

    for i in range(imageSizes[2]):
        single_img = img[:,:,i]

        new_ranges = [ranges[0], ranges[1]]
        scale = max(new_ranges)/min(new_ranges)

        index_min = new_ranges.index(min(new_ranges))

        if index_min == 0:
            height = imageSizes[0]
            width = imageSizes[1] * scale
        else:
            height = imageSizes[0] * scale
            width = imageSizes[1]

        dim = (int(width), int(height))
        print(dim)
        resized = cv2.resize(single_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(fileExport + '\\2\\' + os.path.splitext(filename)[0] + '_' + str(i) + '_2.png', resized)

    for i in range(imageSizes[1]):
        single_img = img[:,i,:]

        new_ranges = [ranges[0], ranges[2]]
        scale = max(new_ranges) / min(new_ranges)

        index_min = new_ranges.index(min(new_ranges))

        if index_min == 0:
            height = imageSizes[0]
            width = imageSizes[2] * scale
        else:
            height = imageSizes[0] * scale
            width = imageSizes[2]

        dim = (int(width), int(height))
        resized = cv2.resize(single_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(fileExport + '\\1\\' + os.path.splitext(filename)[0] + '_' + str(i) + '_1.png', resized)

    for i in range(imageSizes[0]):
        single_img = img[i,:,:]

        new_ranges = [ranges[1], ranges[2]]
        scale = max(new_ranges) / min(new_ranges)

        index_min = new_ranges.index(min(new_ranges))

        if index_min == 0:
            height = imageSizes[1]
            width = imageSizes[2] * scale
        else:
            height = imageSizes[1] * scale
            width = imageSizes[2]

        dim = (int(width), int(height))
        resized = cv2.resize(single_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(fileExport + '\\0\\' + os.path.splitext(filename)[0] + '_' + str(i) + '_0.png', resized)

def ExtractSingleOctFile(folder, folder_export, filename):
    start = time.time()
    print(start.real)
    if not (os.path.exists(folder_export)):
        os.mkdir(folder_export)

    fileExport = folder_export + '\\' + os.path.splitext(filename)[0]
    if (os.path.exists(fileExport)):
        return

    os.mkdir(fileExport)

    print('start unzipping')
    unzippedpath = UnzipOctFile(folder, filename, fileExport)
    print('start read real and save image')
    ReadRealDataAndSaveImage(filename, unzippedpath, fileExport)

    # shutil.rmtree(unzippedpath)

    end = time.time()
    print('time: ')
    print(end - start)
    print('\n')


def ReadOcts(folder, folder_export):

    if not (os.path.exists(folder_export)):
        os.mkdir(folder_export)

    for filename in os.listdir(folder + '\\'):
        ExtractSingleOctFile(folder, folder_export, filename)


def ReadFileListAndExtractOct(fileList, folder, folder_export):
    '''f = open(fileList, "r")
    for line in f:
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        filename = line.partition(';')[0] + '.oct'
        path = r"" + folder + '\\' + filename
        print(path)
        if not(os.path.exists(path)):
            print("fail")
            continue'''
        # f.close()

    oct_files = os.listdir(os.path.join(folder, folder))
    for oct in oct_files:
        if "Vorstudie" in str(oct):
            continue
        ExtractSingleOctFile(folder, folder_export, oct)





#Folder to OCT files
folder = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\Leber\\Parenchym'
#Folder to save image exports
folder_export = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\Training original\\Parenchym'

#ReadFileListAndExtractOct("Z:\\magen_mucosa.txt", folder, folder_export + '\\Magen_Mucosa')
ReadFileListAndExtractOct("C:\\Users\\kgr-ik\\Desktop\\MA_Inna\\data\\list_krlm_CLEANED.txt", folder, folder_export)

