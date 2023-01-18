import matplotlib.pyplot as plt
import numpy as np
import os

xception = 1

def CalcMetricsForArrays(trueScan, predScan):
    tLen = len(trueScan)
    pLen = len(predScan)

    if (tLen != pLen):
        print('error')
        return

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(tLen):
        tItem = trueScan[i]
        pItem = predScan[i]

        if (xception == 1 and pItem > 0.5) or (xception == 0 and pItem > 0.5):
            pItem = 1
        else:
            pItem = 0

        if tItem == 0 and pItem == 0:
            TN = TN + 1
        elif tItem == 0 and pItem == 1:
            FP = FP + 1
        elif tItem == 1 and pItem == 0:
            FN = FN + 1
        elif tItem == 1 and pItem == 1:
            TP = TP + 1

    # print('TP ' + str(TP))
    # print('\n')
    # print('FP ' + str(FP))
    # print('\n')
    # print('TN ' + str(TN))
    # print('\n')
    # print('FN ' + str(FN))
    # print('\n')

    #print(str(TP) + '	' + str(FP))
    #print(str(FN) + '	' + str(TN))

    Recall = TP/(TP+FP + 0.00000000001)
    Precision = TP/(TP+FN+ 0.00000000001)
    F1 = (2*Recall*Precision)/(Recall + Precision + 0.00000000001)

    print(str(F1).replace('.',','))
    #print('\n')

path = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\Results\\CRLM\\200322 first run\\all_20220324_180448\\xception_ckpt4\\'

for resultFile in (os.listdir(path)):

    if 'single' in resultFile:
        continue

    f = open(path + resultFile, 'r')
    #line = ''
    for line in f:
        #line = f.readline()
        if 'D:' in line:
            if 'xception' in line:
                xception = 1
                #print('Xception\n')
            else:
                xception = 0
                #print ('VGG \n')
            #print(line)

        if 'valid all' in line:
            trueBScan = f.readline()
            predBscan = f.readline()

            trueBScan = trueBScan.replace("[", "")
            trueBScan = trueBScan.replace("]", "")
            trueBScan = trueBScan.replace("\n", "")
            trueBScan = trueBScan.replace(" ", "")
            trueBScan = trueBScan.split(',')
            trueBScan.remove('')
            trueBScan = np.uint8(trueBScan)

            predBscan = predBscan.replace("[", "")
            predBscan = predBscan.replace("]", "")
            predBscan = predBscan.replace("\n", "")
            predBscan = predBscan.replace(" ", "")
            predBscan = predBscan.split(',')
            predBscan.remove('')
            predBscan = [float(i) for i in predBscan]
            #CalcMetricsForArrays(trueBScan, predBscan)

        if 'test all' in line:
            trueBScan = f.readline()
            predBscan = f.readline()

            trueBScan = trueBScan.replace("[", "")
            trueBScan = trueBScan.replace("]", "")
            trueBScan = trueBScan.replace("\n", "")
            trueBScan = trueBScan.replace(" ", "")
            trueBScan = trueBScan.split(',')
            trueBScan.remove('')
            trueBScan = np.uint8(trueBScan)

            predBscan = predBscan.replace("[", "")
            predBscan = predBscan.replace("]", "")
            predBscan = predBscan.replace("\n", "")
            predBscan = predBscan.replace(" ", "")
            predBscan = predBscan.split(',')
            predBscan.remove('')
            predBscan = [float(i) for i in predBscan]

            #print('All ' + resultFile + '\n')
            CalcMetricsForArrays(trueBScan, predBscan)

        if 'test bscan' in line:
            trueBScan = f.readline()
            predBscan = f.readline()

            trueBScan = trueBScan.replace("[", "")
            trueBScan = trueBScan.replace("]", "")
            trueBScan = trueBScan.replace("\n", "")
            trueBScan = trueBScan.replace(" ", "")
            trueBScan = trueBScan.split(',')
            trueBScan.remove('')
            trueBScan = np.uint8(trueBScan)

            predBscan = predBscan.replace("[", "")
            predBscan = predBscan.replace("]", "")
            predBscan = predBscan.replace("\n", "")
            predBscan = predBscan.replace(" ", "")
            predBscan = predBscan.split(',')
            predBscan.remove('')
            predBscan = [float(i) for i in predBscan]

            #print('Bscan ' + resultFile + '\n')
            #CalcMetricsForArrays(trueBScan, predBscan)

        if 'test csan' in line:
            trueCScan = f.readline()
            predCscan = f.readline()

            trueCScan = trueCScan.replace("[", "")
            trueCScan = trueCScan.replace("]", "")
            trueCScan = trueCScan.replace("\n", "")
            trueCScan = trueCScan.replace(" ", "")
            trueCScan = trueCScan.split(',')
            trueCScan.remove('')
            trueCScan = np.uint8(trueCScan)

            predCscan = predCscan.replace("[", "")
            predCscan = predCscan.replace("]", "")
            predCscan = predCscan.replace("\n", "")
            predCscan = predCscan.replace(" ", "")
            predCscan = predCscan.split(',')
            predCscan.remove('')
            predCscan = [float(i) for i in predCscan]

            #print('Cscan ' + resultFile + '\n')
            #CalcMetricsForArrays(trueCScan, predCscan)

