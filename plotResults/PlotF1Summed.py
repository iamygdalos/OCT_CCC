import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size = ''

def GetFromFirstRun(folder, fileName):
    Epoch, Train_F1, Train_Loss, Valid_F1, Valid_Loss = 0,0,0,0,0
    for line in open(folder + '\\' + CVRun + '\\' + fileName, 'r'):
        if (line == '\n'):
            continue
        values = [s for s in line.split(',')]

        if (values[0] == ''):
            continue

        Epoch = int(values[0])
        Train_Loss = float(values[1])

        Train_TP_tmp = float(values[3])
        Train_TN_tmp = float(values[4])
        Train_FP_tmp = float(values[5])
        Train_FN_tmp = float(values[6])

        Train_recall = Train_TP_tmp / (Train_TP_tmp + Train_FN_tmp + 0.0000001)
        Train_precision = Train_TP_tmp / (Train_TP_tmp + Train_FP_tmp + 0.0000001)
        Train_F1 = 2 * Train_recall * Train_precision / (Train_recall + Train_precision + 0.0000001)

        Valid_Loss = float(values[7])

        Val_TP_tmp = float(values[9])
        Val_TN_tmp = float(values[10])
        Val_FP_tmp = float(values[11])
        Val_FN_tmp = float(values[12])

        Val_recall = Val_TP_tmp / (Val_TP_tmp + Val_FN_tmp + 0.0000001)
        Val_precision = Val_TP_tmp / (Val_TP_tmp + Val_FP_tmp + 0.0000001)
        Valid_F1 = 2 * Val_recall * Val_precision / (Val_recall + Val_precision + 0.0000001)

    return Epoch, Train_F1, Train_Loss, Valid_F1, Valid_Loss


def ReadALlFilesForCVRun(folder, CVRun):
    if not (os.path.exists(folder + '\\' + CVRun + '\\plots\\')):
        os.mkdir(folder + '\\' + CVRun + '\\plots\\')

    if not (os.path.exists(folder + '\\' + CVRun + '\\plots\\f1_score\\')):
        os.mkdir(folder + '\\' + CVRun + '\\plots\\f1_score\\')

    colorsTrain = ['lightsteelblue', 'cornflowerblue', 'midnightblue', 'blue', 'slategrey']
    colors = ['crimson', 'orange', 'lime', 'indigo', 'darkred']
    i = 0
    E_FT, train_f1_FT, train_loss_ft, valid_f1_FT, valid_loss_FT = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [
        0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    for fileName in os.listdir(folder + '\\' + CVRun + '\\'):
        if os.path.splitext(fileName)[1] != '.csv':
            continue

        if 'Begin' in fileName:
            E_FT_tmp, train_f1_FT_tmp, train_loss_ft_tmp, valid_f1_FT_tmp, valid_loss_FT_tmp = GetFromFirstRun(folder,
                                                                                                               fileName)

            fSplits = fileName.split('_')
            Nr = fSplits[len(fSplits) - 1]
            Nr = Nr.replace('.csv', '')
            Nr = int(Nr)

            E_FT[Nr] = (E_FT_tmp)
            train_f1_FT[Nr] = (train_f1_FT_tmp)
            train_loss_ft[Nr] = (train_loss_ft_tmp)
            valid_f1_FT[Nr] = (valid_f1_FT_tmp)
            valid_loss_FT[Nr] = (valid_loss_FT_tmp)
            continue

    for fileName in os.listdir(folder + '\\' + CVRun + '\\'):
        Epoch = []
        Train_Loss, Train_BinAcc, Train_TP, Train_TN, Train_FP, Train_FN, Train_Recall, Train_Precision, Train_F1 = [], [], [], [], [], [], [], [], []
        Val_Loss, Val_BinAcc, Val_TP, Val_TN, Val_FP, Val_FN, Val_Recall, Val_Precision, Val_F1 = [], [], [], [], [], [], [], [], []

        if os.path.splitext(fileName)[1] != '.csv':
            continue

        if 'Begin' in fileName:
            continue

        print(fileName)

        fSplits = fileName.split('_')
        Nr = fSplits[len(fSplits) - 1]
        Nr = Nr.replace('.csv', '')
        Nr = int(Nr)

        Epoch.append(E_FT[Nr])
        Train_F1.append(train_f1_FT[Nr])
        Train_Loss.append(train_loss_ft[Nr])
        Val_F1.append(valid_f1_FT[Nr])
        Val_Loss.append(valid_loss_FT[Nr])

        for line in open(folder + '\\' + CVRun + '\\' + fileName, 'r'):
            if(line == '\n'):
                continue
            values = [s for s in line.split(',')]

            if(values[0] == ''):
                continue

            Epoch.append(int(values[0])+1)
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
        #df_b = pd.DataFrame({'x': Epoch, 'y1': Train_BinAcc, 'y2': Val_BinAcc})

        axes = plt.gca()
        axes.set_ylim(bottom=0.0, top=1.1)
        df_f.plot(x='x', y='y1', color=colorsTrain[i], linewidth=1.5, label='train f1-score CV ' + str(i), ax=axes, legend=False)
        df_f.plot(x='x', y='y2', color=colors[i], linewidth=1.5, label='valid f1-score CV ' + str(i), ax=axes, legend=False)
        #p1, = plt.plot(data=df_f, x='x', y='y1',color=colorsTrain[i], linewidth=1.5, label='train f1-score CV ' + str(i))
        #p2, = plt.plot(data=df_f, x='x', y='y2', color=colors[i], linewidth=1.5, label='valid f1-score CV ' + str(i))
        i = i + 1

    #plt.legend(handles=[p1, p2], title='title', bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend(loc='lower left')
    plt.grid()
    plt.ylabel("F1-Score",
               family='arial',
               color='black',
               weight='normal',
               size=12,
               labelpad=1)
    fig = plt.gcf()

    fig.set_size_inches(5.5,2.5)

    plt.savefig(folder + '\\' + CVRun + '\\plots\\f1_score\\' + os.path.splitext(fileName)[0] + '_' + size + '_ALLOVERCV_f1_score.png', bbox_inches='tight', pad_inches=0.1)

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