from LoadXceptionModelAndPredict import LoadXceptionAndPredict
from LoadVGGModelAndPredict import LoadVGGAndPredict
import os

pathSplits = 'Z:\\MA_Inna\\training results\\lulu net\\061221_single_x_view_prepro\\'
modelsPath = 'Z:\\MA_Inna\\training results\\lulu net\\061221_single_x_view_prepro\\'
#pathToModel = 'D:\\MA_Luisa\\models\\MA_Final\\single_20210525_122531\\xception_ckpt1\\xception_MaskedSquaredSingle_1_1'

imgPath = 'Z:\\MA_Inna\\data\\sepbyviewsplitted\\xview\\train'

trainfile = "C:\\Users\\kgr-ik\\Desktop\\MA_Inna\\data\\list_krlm_TRAIN.txt"
testfile = "C:\\Users\\kgr-ik\\Desktop\\MA_Inna\\data\\list_krlm_TEST.txt"


validSet = []
for CVRun in os.listdir(pathSplits):
    if len(validSet) > 3:
        break
    if 'cv_' in CVRun:
        path = os.path.join(pathSplits, CVRun)
        validfile = open(path, 'r')
        lines = validfile.readlines()
        tmp = []
        for line in lines[1].split():
            line = line.partition(']')[0]
            if line.isdigit():
                tmp.append(int(line))
        validSet.append(tmp)


for CVRun in os.listdir(modelsPath):
    modelfile = modelsPath + CVRun + '.txt'
    with open(modelfile, mode='w') as f:
        for modelFolder in os.listdir(modelsPath + CVRun):
            if 'ckpt' in modelFolder:
                for ckptFile in os.listdir(modelsPath + CVRun + '\\' + modelFolder):
                    if '.index' in ckptFile:

                        pathToModel = modelsPath + CVRun + '\\' + modelFolder + '\\' + os.path.splitext(ckptFile)[0]
                        parts = os.path.splitext(ckptFile)[0].split('_')

                        cvIndex = int(parts[6])

                        dataPath = imgPath

                        validset = validSet[cvIndex]

                        f.write('\n')
                        f.write(pathToModel)
                        f.write('\n')

                        if parts[0] == 'xception':
                            validAllTrueTrainLabels, validPredictionsVal, validAllTrueTrainLabelsBscan, validPredictionsValBscan, validAllTrueTrainLabelsCscan, validPredictionsValCscan, testAllTrueTrainLabels, testPredictionsVal, testAllTrueTrainLabelsBscan, testPredictionsValBscan, testAllTrueTrainLabelsCscan, testPredictionsValCscan \
                                = LoadXceptionAndPredict(pathToModel, dataPath, validset, trainfile, testfile)

                            f.write('\n')
                            f.write('valid all \n')
                            f.write('[')
                            for item in validAllTrueTrainLabels:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validAllTrueTrainLabels)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsVal:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validPredictionsVal)

                            f.write('\n')
                            f.write('valid bscan \n')
                            f.write('[')
                            for item in validAllTrueTrainLabelsBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsValBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('valid cscan \n')
                            f.write('[')
                            for item in validAllTrueTrainLabelsCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsValCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('test all \n')
                            f.write('[')
                            for item in testAllTrueTrainLabels:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testAllTrueTrainLabels)
                            f.write('\n')
                            for item in testPredictionsVal:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')

                            #f.write(testPredictionsVal)
                            f.write('\n')
                            f.write('test bscan \n')
                            f.write('[')
                            for item in testAllTrueTrainLabelsBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in testPredictionsValBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('test csan \n')
                            f.write('[')
                            for item in testAllTrueTrainLabelsCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in testPredictionsValCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testPredictionsValCscan)

                        elif parts[0] == 'vgg':
                            validAllTrueTrainLabels, validPredictionsVal, validAllTrueTrainLabelsBscan, validPredictionsValBscan, validAllTrueTrainLabelsCscan, validPredictionsValCscan, testAllTrueTrainLabels, testPredictionsVal, testAllTrueTrainLabelsBscan, testPredictionsValBscan, testAllTrueTrainLabelsCscan, testPredictionsValCscan = LoadVGGAndPredict(pathToModel, dataPath, validset)

                            f.write('\n')
                            f.write('valid all \n')
                            f.write('[')
                            for item in validAllTrueTrainLabels:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validAllTrueTrainLabels)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsVal:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validPredictionsVal)

                            f.write('\n')
                            f.write('valid bscan \n')
                            f.write('[')
                            for item in validAllTrueTrainLabelsBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsValBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('valid cscan \n')
                            f.write('[')
                            for item in validAllTrueTrainLabelsCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in validPredictionsValCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('test all \n')
                            f.write('[')
                            for item in testAllTrueTrainLabels:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testAllTrueTrainLabels)
                            f.write('\n')
                            for item in testPredictionsVal:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testPredictionsVal)

                            #f.write(testPredictionsVal)
                            f.write('\n')
                            f.write('test bscan \n')
                            f.write('[')
                            for item in testAllTrueTrainLabelsBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in testPredictionsValBscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            # f.write(validPredictionsValCscan)

                            f.write('\n')
                            f.write('test csan \n')
                            f.write('[')
                            for item in testAllTrueTrainLabelsCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testAllTrueTrainLabelsCscan)
                            f.write('\n')
                            f.write('[')
                            for item in testPredictionsValCscan:
                                f.write(str(item))
                                f.write(', ')
                            f.write(']')
                            #f.write(testPredictionsValCscan)

    f.close()

#LoadXceptionAndPredict(pathToModel, imgPath, validset)