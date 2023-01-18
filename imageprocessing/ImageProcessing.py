import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

imgNum = 2


# removes outliers/artifacts from images
def OutlierRemovalBasedOnMean(img):
    overall_mean = np.mean(img)
    means = np.mean(img, axis=0)
    indices = np.where(means > (overall_mean + 7))
    CorrectedSection = img
    CorrectedSection = cv2.cvtColor(CorrectedSection, cv2.COLOR_GRAY2BGR)
    #CorrectedSection[:, indices] = [0, 255, 0]
    #CorrectedSection[:, indices] = 0
    CorrectedSection = np.delete(CorrectedSection, np.where(np.sum(CorrectedSection, axis=0) == 0)[0], axis=1)
    return CorrectedSection


def MedianBlur(img):
    img2 = cv2.medianBlur(img, 5)
    return img2


def Threshold(img):
    ret, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    return thresh


def FloodFill(img):
    img2 = img.copy()
    h, w = img.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img2, mask, (w - 1, h - 1), 255)

    return img2


# finds black objects (holes) in image and removes them/fills them white
def FillBlackHoles(img):
    imgCopy = img.copy()
    imgCopy = cv2.bitwise_not(imgCopy)

    contours, hier = cv2.findContours(imgCopy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cSorted = sorted(contours, key=cv2.contourArea)

    if (len(cSorted) > 2):
        cSorted = cSorted[0:len(cSorted) - 2]

        for c in cSorted:
            cv2.drawContours(imgCopy, [c], -1, 0, thickness=-1)

    # imgCopyColor = img.copy()
    # imgCopyColor = cv2.cvtColor(imgCopyColor, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(imgCopyColor, contours, -1, (0,255,0), 3)

    imgCopy = cv2.bitwise_not(imgCopy)
    return imgCopy


def FindEges(img):
    edges = cv2.Canny(img, 20, 50)

    return edges


# finds last white pixel in image, draws a horizontal line on this point through the image and performs floodfill operation from this line to the upper part to cover
# tissue information in white
def FindLastWhitePixel(img):
    imgCopy = img.copy()

    indx = np.where(np.sum(imgCopy, axis=1) != 0)[0]

    lastIdx = indx[len(indx) - 1]

    imgCopy[lastIdx, :] = 255

    h, w = imgCopy.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    X = int(np.floor(w / 2))

    # imgCopy[lastIdx + 60, int(np.floor(w/2))] = 255

    steps = int(np.floor(w / 10))
    for i in range(10):
        cv2.floodFill(imgCopy, mask, (steps * i, lastIdx - 2), 255)

    return imgCopy


# finds white objects in image and removes them except the biggest one (-->tissue, rest is noise)
def FillWhiteHoles(img):
    imgCopy = img.copy()

    contours, hier = cv2.findContours(imgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 1):
        cSorted = sorted(contours, key=cv2.contourArea)

        cSorted = cSorted[0:len(cSorted) - 1]

        for c in cSorted:
            cv2.drawContours(imgCopy, [c], -1, 0, thickness=-1)

    # imgCopyColor = imgCopy.copy()
    # imgCopyColor = cv2.cvtColor(imgCopyColor, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(imgCopyColor, contours, -1, (0,255,0), 3)

    return imgCopy


# removes small white objects from image (noise)
def RemoveSmallObjects(img):
    imgCopy = img.copy()

    contours, hier = cv2.findContours(imgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_FLOODFILL)

    if (len(contours) > 1):
        cSorted = sorted(contours, key=cv2.contourArea)

        cSorted = cSorted[0:len(cSorted) - 1]

        for c in cSorted:
            if (cv2.contourArea(c) < 200):
                cv2.drawContours(imgCopy, [c], -1, 0, thickness=-1)

    return imgCopy


# find gaps in tissue surface and close them
def FindGapsAndConnect(img):
    imgCopy = img.copy()
    height, width = imgCopy.shape

    lastWhites = []
    for col in range(width):
        indx = np.where(imgCopy[:, col] != 0)[0]

        for i in range(len(indx) - 2):
            if ((indx[i] + 1) is not indx[i + 1]):
                colorI = 0
                while ((indx[i] + colorI) < indx[i + 1]):
                    if (indx[i] + colorI) < height:
                        imgCopy[indx[i] + colorI, col] = 255
                    colorI = colorI + 1

        indx = np.where(imgCopy[:, col] != 0)[0]
        allgewd = False

        if col == 0:
            lastWhites = indx
        else:
            for lastIndx in lastWhites:
                if lastIndx in indx:
                    allgewd = True
                    break

            if not allgewd:
                if len(indx) == 0:
                    imgCopy[lastWhites, col] = 255
                else:
                    minIdx = np.min(indx)
                    maxIdx = np.max(indx)

                    minLastIdx = np.min(lastWhites)
                    maxLastIdx = np.max(lastWhites)

                    if minLastIdx > minIdx and maxLastIdx > minIdx:
                        # previous below current col

                        dist = minLastIdx - maxIdx
                        imgCopy[maxIdx - 1, col] = 255
                        imgCopy[maxIdx, col] = 255
                        for i in range(dist):
                            imgCopy[maxIdx + i, col] = 255

                    elif minLastIdx < minIdx and maxLastIdx < minIdx:
                        # previous above current col

                        dist = minIdx - maxLastIdx
                        imgCopy[maxLastIdx - 1, col] = 255
                        imgCopy[maxLastIdx, col] = 255
                        for i in range(dist):
                            imgCopy[maxLastIdx + i, col] = 255
                    else:
                        # previous between
                        i = 0

            lastWhites = indx

    return imgCopy


# pipeline of getting a tissue mask
def DoAll(img):
    imgCopy = img.copy()
    img3 = MedianBlur(imgCopy)

    img4 = Threshold(img3)

    img4 = RemoveSmallObjects(img4)
    toDelete = np.where(np.sum(img4, axis=0) == 0)[0]
    img4 = np.delete(img4, np.where(np.sum(img4, axis=0) == 0)[0], axis=1)
    img4 = FindGapsAndConnect(img4)

    kernel = np.ones((3, 3), np.uint8)
    img5 = cv2.dilate(img4, kernel, iterations=1)
    img5 = cv2.erode(img5, kernel, iterations=1)

    img6 = FindLastWhitePixel(img5)

    img7 = FillBlackHoles(img6)
    img8 = FillWhiteHoles(img7)

    return [img8, toDelete]


def ApplyMask(img, mask):
    res = cv2.bitwise_and(img, img, mask=mask)

    return res


# perform k Means on single image and color found clusters
def KMeans(imgInput):
    k = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    img_pixel_values = imgInput.reshape((-1, 1))
    # convert to float
    img_pixel_values = np.float32(img_pixel_values)
    _, h_labels, (h_centers) = cv2.kmeans(img_pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    h_centers = np.uint8(h_centers)
    # flatten the labels array
    h_labels = h_labels.flatten()
    # convert all pixels to the color of the centroids
    h_segmented_image = h_centers[h_labels.flatten()]
    # reshape back to the original image dimension
    t = h_labels.flatten().reshape(imgInput.shape)
    h_segmented_image = h_segmented_image.reshape(imgInput.shape)
    # show the image

    colors = np.unique(h_segmented_image)

    kMeansImg = np.uint8(h_segmented_image)
    kMeansImg = cv2.cvtColor(kMeansImg, cv2.COLOR_GRAY2BGR)

    i = 0
    newColors = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 255, 0]]
    for c in colors:
        indices0 = np.where(kMeansImg == [c, c, c])
        kMeansImg[indices0[0], indices0[1], :] = newColors[i]
        i = i + 1

    return kMeansImg


def DeleteRows(img):
    CorrectedSection = np.delete(img, np.where(np.sum(img, axis=1) == 0)[0], axis=0)
    return CorrectedSection


##applies tissue mask to image and crop a square sized part from the middle of the image
def ApplyMaskAndDoMiddleCropSquare(base_folder, base_output):
    if not (os.path.exists(base_output)):
        os.mkdir(base_output)

    for diagnoseDir in os.listdir(base_folder):

        if not (os.path.exists(base_output + '\\' + diagnoseDir)):
            os.mkdir(base_output + '\\' + diagnoseDir)

        for patient in os.listdir(base_folder + '\\' + diagnoseDir):
            print(patient)

            if not (os.path.exists(base_output + '\\' + diagnoseDir + '\\' + patient)):
                os.mkdir(base_output + '\\' + diagnoseDir + '\\' + patient)

            if not (os.path.exists(base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\')):
                os.mkdir(base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\')

            for fileName in os.listdir(base_folder + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\'):
                # print(fileName)

                img = cv2.imread(
                    base_folder + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                    cv2.IMREAD_GRAYSCALE)
                img = OutlierRemovalBasedOnMean(img)
                [mask, toDelete] = DoAll(img)
                img = np.delete(img, toDelete, axis=1)
                res = ApplyMask(img, mask)

                width, height = res.shape

                if width == height:
                    cv2.imwrite(
                        base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName, res)
                elif width < height:
                    diff = height - width
                    idx = np.int32(np.floor(diff / 2))
                    imgCropped = res[idx:idx + width, :]
                    cv2.imwrite(
                        base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                        imgCropped)
                else:
                    diff = width - height
                    idx = np.int32(np.floor(diff / 2))
                    imgCropped = res[:, idx:idx + height]
                    cv2.imwrite(
                        base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                        imgCropped)


##applies tissue mask to image and create overlapping squaresized images from original image
def ApplyMaskAndDoOverlappingSquares(base_folder, base_output):
    if not (os.path.exists(base_output)):
        os.mkdir(base_output)

    for patient in os.listdir(base_folder):
        if 'temp' in patient:
            continue
        print(patient)
        if not (os.path.exists(os.path.join(base_output, patient))):
            os.mkdir(os.path.join(base_output, patient))

        for view in os.listdir(os.path.join(base_folder, patient)):
            if view == '0':
                continue
            if not (os.path.exists(os.path.join(base_output, patient,  str(view)))):
                os.mkdir(os.path.join(base_output, patient,  str(view)))
            else:
                continue

            for fileName in os.listdir(os.path.join(base_folder, patient, str(view))):
                # print(fileName)

                img = cv2.imread(os.path.join(base_folder, patient, str(view), fileName), cv2.IMREAD_GRAYSCALE)
                img = OutlierRemovalBasedOnMean(img)
                [mask, toDelete] = DoAll(img)
                img = np.delete(img, toDelete, axis=1)
                res = ApplyMask(img, mask)
                res = DeleteRows(res)

                # cv2.namedWindow('win', cv2.WINDOW_NORMAL)
                # cv2.imshow('win', mask)
                # cv2.waitKey(0)

                height, width = res.shape
                min_shape = min(width, height)
                max_shape = max(width, height)
                imgCount = int(np.round(max_shape / min_shape))
                overlap = (min_shape - (max_shape - min_shape * imgCount)) / imgCount

                for i in range(imgCount + 1):
                    startingPoint = int(np.floor((i * min_shape) - (i * overlap)))
                    filename = fileName.partition('.png')[0] + '_' + str(i) + '.png'
                    if max_shape == width:
                        imgCropped = res[0:min_shape, startingPoint:(startingPoint + min_shape)]
                    else:
                        imgCropped = res[startingPoint:(startingPoint + min_shape), 0:min_shape]

                    imgCropped = cv2.cvtColor(imgCropped, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join(base_output, patient, str(view), filename), imgCropped)
    print('DONE')

##applies tissue mask to image and save only cropped rectangle (tissue information)
def ApplyMaskAndSaveCroppedRectangular(base_folder, base_output):
    if not (os.path.exists(base_output)):
        os.mkdir(base_output)

    for diagnoseDir in os.listdir(base_folder):

        if not (os.path.exists(base_output + '\\' + diagnoseDir)):
            os.mkdir(base_output + '\\' + diagnoseDir)

        for patient in os.listdir(base_folder + '\\' + diagnoseDir):
            print(patient)

            if not (os.path.exists(base_output + '\\' + diagnoseDir + '\\' + patient)):
                os.mkdir(base_output + '\\' + diagnoseDir + '\\' + patient)

            if not (os.path.exists(base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\')):
                os.mkdir(base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\')

            for fileName in os.listdir(base_folder + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\'):
                # print(fileName)

                img = cv2.imread(
                    base_folder + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                    cv2.IMREAD_GRAYSCALE)
                img = OutlierRemovalBasedOnMean(img)
                mask = DoAll(img)
                res = ApplyMask(img, mask)
                res = DeleteRows(res)

                cv2.imwrite(base_output + '\\' + diagnoseDir + '\\' + patient + '\\' + str(imgNum) + '\\' + fileName,
                            res)


##applies tissue mask to image and save in original size
def ApplyMaskAndSaveRealSize(base_folder, base_output):
    if not (os.path.exists(base_output)):
        os.mkdir(base_output)

    for patient in os.listdir(base_folder):
        if 'temp' in patient:
            continue
        print(patient)
        if not (os.path.exists(os.path.join(base_output, patient))):
            os.mkdir(os.path.join(base_output, patient))

        for view in os.listdir(os.path.join(base_folder, patient)):
            if view == '0':
                continue
            if not (os.path.exists(os.path.join(base_output, patient,  str(view)))):
                os.mkdir(os.path.join(base_output, patient,  str(view)))

            for fileName in os.listdir(os.path.join(base_folder, patient, str(view))):
                # print(fileName)
                file_path = os.path.join(base_folder, patient, str(view), fileName)
                output_path = os.path.join(base_output, patient, str(view), fileName)
                if (os.path.exists(output_path)):
                    continue
                img = img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = OutlierRemovalBasedOnMean(img)
                [mask, toDelete] = DoAll(img)
                img = np.delete(img, toDelete, axis=1)
                res = ApplyMask(img, mask)
                cv2.imwrite(output_path, res)


# base_folder = 'Z:\\OCT_Data_Export\\'
# base_output = 'Z:\\OCT_Data_Export_Processed\\'
# imgNum = 2
# ApplyMaskAndDoOverlappingSquares(base_folder, base_output)

# base_folder = 'D:\\MA_Luisa\\data\\exportKRLM_1_Half\\'
# base_output = 'D:\\MA_Luisa\\data\\exportKRLM_1_Half_MaskedSquared\\'
# imgNum = 1
# ApplyMaskAndDoSquare(base_folder, base_output)

folder_list = ['HCC']
for folder in folder_list:
    base_folder = os.path.join('F:\\OCT-Dateien\\Training original', folder)
    base_output = os.path.join('C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\Training preprocessed cropped', folder)
    #ApplyMaskAndSaveRealSize(base_folder, base_output)
    ApplyMaskAndDoOverlappingSquares(base_folder, base_output)
