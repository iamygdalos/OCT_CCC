import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ReadALlFilesForCVRun(folder, CVRun):
    if not (os.path.exists(folder + '\\' + CVRun + '\\plots\\')):
        os.mkdir(folder + '\\' + CVRun + '\\plots\\')

    if not (os.path.exists(folder + '\\' + CVRun + '\\plots\\losses\\')):
        os.mkdir(folder + '\\' + CVRun + '\\plots\\losses\\')

    for fileName in os.listdir(folder + '\\' + CVRun + '\\'):
        Epoch = []
        Train_Loss, Train_BinAcc, Train_TP, Train_TN, Train_FP, Train_FN, Train_Recall, Train_Precision, Train_F1 = [], [], [], [], [], [], [], [], []
        Val_Loss, Val_BinAcc, Val_TP, Val_TN, Val_FP, Val_FN, Val_Recall, Val_Precision, Val_F1 = [], [], [], [], [], [], [], [], []

        if os.path.splitext(fileName)[1] != '.csv':
            continue

        if 'Begin' in fileName:
            continue

        print(fileName)

        for line in open(folder + '\\' + CVRun + '\\' + fileName, 'r'):
            if(line == '\n'):
                continue
            values = [s for s in line.split(',')]

            if(values[0] == ''):
                continue

            Epoch.append(int(values[0]))
            Train_Loss.append(float(values[1]))
            Train_BinAcc.append(float(values[2]))
            Train_TP.append(float(values[3]))
            Train_TN.append(float(values[4]))
            Train_FP.append(float(values[5]))
            Train_FN.append(float(values[6]))

            Train_TP_tmp = float(values[3])
            Train_TN_tmp = float(values[4])
            Train_FP_tmp = float(values[5])
            Train_FN_tmp = float(values[6])

            Train_recall = Train_TP_tmp / (Train_TP_tmp + Train_FN_tmp + 0.0000001)
            Train_precision = Train_TP_tmp / (Train_TP_tmp + Train_FP_tmp + 0.0000001)
            Train_F1_tmp = 2*Train_recall*Train_precision / (Train_recall+Train_precision + 0.0000001)

            Train_Recall.append(Train_recall)
            Train_Precision.append(Train_precision)
            Train_F1.append(Train_F1_tmp)

            Val_Loss.append(float(values[7]))
            Val_BinAcc.append(float(values[8]))
            Val_TP.append(float(values[9]))
            Val_TN.append(float(values[10]))
            Val_FP.append(float(values[11]))
            Val_FN.append(float(values[12]))

            Val_TP_tmp = float(values[9])
            Val_TN_tmp = float(values[10])
            Val_FP_tmp = float(values[11])
            Val_FN_tmp = float(values[12])

            Val_recall = Val_TP_tmp / (Val_TP_tmp + Val_FN_tmp + 0.0000001)
            Val_precision = Val_TP_tmp / (Val_TP_tmp + Val_FP_tmp + 0.0000001)
            Val_F1_tmp = 2 * Val_recall * Val_precision / (Val_recall + Val_precision + 0.0000001)

            Val_Recall.append(Val_recall)
            Val_Precision.append(Val_precision)
            Val_F1.append(Val_F1_tmp)

        #print(fileName)
        #print(Train_F1[len(Train_F1)-1])
        #print(Val_F1[len(Val_F1)-1])


        df = pd.DataFrame({'x': Epoch, 'y1': Train_Loss, 'y2': Val_Loss})
        df_f = pd.DataFrame({'x': Epoch, 'y1': Train_F1, 'y2': Val_F1})
        df_b = pd.DataFrame({'x': Epoch, 'y1': Train_BinAcc, 'y2': Val_BinAcc})

        axes = plt.gca()
        axes.set_ylim(bottom=0.0, top=10.0)
        plt.plot('x', 'y1', data=df, color='darkblue', linewidth=1.5, label='train loss')
        plt.plot('x', 'y2', data=df, color='crimson', linewidth=1.5, label='valid loss')

        plt.legend()
        plt.grid()

        fig = plt.gcf()

        fig.set_size_inches(3.0,3.0)

        plt.savefig(folder + '\\' + CVRun + '\\plots\\losses\\' + os.path.splitext(fileName)[0] + '_losses.png')

        plt.clf()
        plt.close(fig)

#folderModels - path to directory where result lies like:
# //someModelname//xception_historyBegin_Magen_20_0.csv
# //someModelname//xception_finalmetrics_Magen_20.txt
# //someModelname//xception_metricOnTest_Magen_20_2.txt

folderModels = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\Results\\HCC\\130422 HCC 1st run\\'

for CVRun in os.listdir(folderModels):
    if '.txt' in CVRun:
        continue
    if '.csv' in CVRun:
        continue
    ReadALlFilesForCVRun(folderModels, CVRun)