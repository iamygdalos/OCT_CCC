import zipfile
import os
import cv2
import numpy as np
import time
import xml.etree.ElementTree as ET
import shutil

#ReadOcts() reads .oct file from folder, scales the intensity based on histogram information and saves them to output folder

def UnzipOctFile(folder, filename, folder_export):
    tempDir = folder_export + 'temp'
    if (os.path.exists(tempDir)):
        return tempDir

    os.mkdir(tempDir)

    with zipfile.ZipFile(folder + filename, 'r') as zip_ref:
        zip_ref.extractall(tempDir)

    return tempDir

def GetImageSizes(filepath):
    xmlPath = filepath + '\\Header.xml'

    tree = ET.parse(xmlPath)
    root = tree.getroot()

    imageSizes = [int(root[0][2].attrib['SizeZ']), int(root[0][2].attrib['SizeX']),
                          int(root[0][2].attrib['SizeY'])]

    ranges = [float(root[0][2].attrib['RangeZ']), float(root[0][2].attrib['RangeX']),
                          float(root[0][2].attrib['RangeY'])]

    return imageSizes, ranges


def ReadRealDataAndSaveImage(filename, filepath, fileExport, infofile):
    imageSizes, ranges = GetImageSizes(filepath)

    infofile.write(", x: " + str(imageSizes[1]) + ", y: " + str(imageSizes[2]) + ", z: " + str(
        imageSizes[0]))

    dataPath = filepath + '\\data\\Intensity.data'

    img = np.zeros((imageSizes[0], imageSizes[1], imageSizes[2]))

    with open(dataPath, 'r') as f:
        raw = np.fromfile(f, np.float32, imageSizes[0]*imageSizes[1]*imageSizes[2])

    data = raw.reshape((imageSizes[0], imageSizes[1], imageSizes[2]), order='F')
    data = data.astype(int)
    data[data < 0] = 0

    max_data = np.max(data)

    half = imageSizes[2]/2
    hist, bin_edges = np.histogram(data[:,:,int(half)], bins=max_data)  # arguments are passed to np.histogram
    avg_hist = sum(hist)/len(hist)
    hist[hist < avg_hist] = 0
    non_zero = np.nonzero(hist)

    max_data = non_zero[0][len(non_zero[0])-1]

    data[data > max_data] = max_data

    img = (255.0 / max_data) * data

    if not (os.path.exists(fileExport + '\\2\\')):
        os.mkdir(fileExport + '\\2\\')

    if not (os.path.exists(fileExport + '\\1\\')):
        os.mkdir(fileExport + '\\1\\')

    if not (os.path.exists(fileExport + '\\0\\')):
        os.mkdir(fileExport + '\\0\\')

    #save all images
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

    if not (os.path.exists(folder_export)):
        os.mkdir(folder_export)

    fileExport = folder_export + '\\' + os.path.splitext(filename)[0]
    if (os.path.exists(fileExport)):
        return

    os.mkdir(fileExport)

    unzippedpath = UnzipOctFile(folder, filename, fileExport)
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
    f = open(fileList, "r")
    for line in f:
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        if not(os.path.exists(folder + '\\' + line)):
            continue

        ExtractSingleOctFile(folder, folder_export, line)

    f.close()

#Folder to OCT files
folder = 'Z:\\OCT_Data_1_Backup'
#Folder to save image exports
folder_export = 'Z:\\OCT_Data_Export'

#ReadFileListAndExtractOct("Z:\\magen_mucosa.txt", folder, folder_export + '\\Magen_Mucosa')
ReadFileListAndExtractOct("Z:\\magen_tumor.txt", folder, folder_export + '\\Magen_Tumor')
