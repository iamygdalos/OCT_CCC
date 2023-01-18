import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from training.TrainOnModel import keras_train_Xception_AllData
from training.TrainOnModel import keras_train_VGG16_AllData
import time
import tensorflow as tf

tf.keras.backend.clear_session()

# setting the seed for reproducible results
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

patients = []
labels = []
f = open("C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\list_HCC_Train_SchnelleAnalyse.txt", "r")
for l in f:
    splits = l.split(';')
    patients.append(splits[0])
    labels.append(splits[1])
f.close()

testPatients = []
testLabels = []
f = open("C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\list_HCC_Test_SchnelleAnalyse.txt", "r")
for l in f:
    splits = l.split(';')
    testPatients.append(splits[0])
    testLabels.append(splits[1])
f.close()

# random.shuffle(patients)
availableLabels = ['normal', 'abnormal']
source_path = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\OCT-Dateien\\Training preprocessed cropped\\'
result_foldername = 'C:\\Users\\marti\\Desktop\\OCT Liver Classification\\Results\\'

result_name = '110522 HCC 5th run'
tumor = 'HCC'

result_foldername = os.path.join(result_foldername, tumor)
if not os.path.exists(result_foldername):
    os.mkdir(result_foldername)

result_foldername = os.path.join(result_foldername, result_name)
if not os.path.exists(result_foldername):
    os.mkdir(result_foldername)
i=1
# for i in range(5):
skf = StratifiedKFold(n_splits=5, shuffle=True)
skfSplits = []
cvRun = 0
for train, testIdx in skf.split(np.array(patients), labels):
    with open(result_foldername + 'cv_' + str(i) + '_run_' + str(cvRun) + '.txt', mode='w') as f:
        f.write('train: ')
        f.write(str(train))
        f.write('\n valid: ')
        f.write(str(testIdx))

    skfSplits.append([train, testIdx])

    cvRun = cvRun + 1

resultsFolderName = result_foldername + '\\all_' + time.strftime("%Y%m%d_%H%M%S")
keras_train_Xception_AllData.Training(source_path, skfSplits, availableLabels, patients, labels, testPatients,
                                      testLabels, resultsFolderName, result_name, batchSize=9, epochNum=10, tumortype=tumor)
