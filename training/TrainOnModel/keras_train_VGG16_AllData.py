import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import random

def GetImageFiles(dirPath):
    return list(Path(dirPath).rglob('*.[pP][nN][gG]'))

def Training(imgPath, skfSplits, patients, labels, patientsTest, labelsTest, name, resultFolderName, batchSize, epochNum):
    if not (os.path.exists(resultFolderName)):
        os.mkdir(resultFolderName)

    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    datagen = ImageDataGenerator(rescale=1 / 255)

    test = pd.DataFrame(columns=('filename', 'class'))
    i = 0
    for pat in patientsTest:
        lab = labelsTest[i]
        i = i + 1
        patPath = imgPath

        if lab == "T":
            patPath = patPath + "abnormal\\"
        else:
            patPath = patPath + "normal\\"

        patPath = patPath + pat

        tmp = GetImageFiles(patPath)
        print('test path:' + patPath)
        for t in tmp:
            #testPatients = tf.concat(testPatients, str(t), axis=0)
            #testLabels = tf.concat(testLabels, str(lab), axis=0)
            values_to_add = {'filename': str(t), 'class': str(lab)}
            row_to_add = pd.Series(values_to_add, name=t)
            test = test.append(row_to_add, ignore_index=True)
    #test_ds = tf.data.Dataset.from_tensor_slices((testPatients, testLabels))
    #print(test_ds.element_spec)

    m = 0
    tst_acc = []
    tst_f1 = []
    for s in skfSplits:
        train = s[0]
        testIdx = s[1]

        tr = pd.DataFrame(columns=('filename', 'class'))
        valid = pd.DataFrame(columns=('filename', 'class'))
        #trainPatients = tf.zeros([len(train), 1])
        #vaildPatients = tf.zeros([len(testIdx), 1])

        for i in train:
            pat = patients[i]
            lab = labels[i]
            patPath = imgPath

            if lab == "T":
                patPath = patPath + "abnormal\\"
            else:
                patPath = patPath + "normal\\"

            patPath = patPath + pat
            tmp = GetImageFiles(patPath)
            print('train path:'+patPath)

            for t in tmp:
                values_to_add = {'filename': str(t), 'class': str(lab)}
                row_to_add = pd.Series(values_to_add, name=t)
                tr = tr.append(row_to_add, ignore_index=True)

        for i in testIdx:
            pat = patients[i]
            lab = labels[i]
            patPath = imgPath

            if lab == "T":
                patPath = patPath + "abnormal\\"
            else:
                patPath = patPath + "normal\\"
            patPath = patPath + pat
            print('valid path:'+patPath)
            tmp = GetImageFiles(patPath)
            #valid = pd.concat([pd.DataFrame([t, lab], columns=['filename', 'class']) for t in tmp], ignore_index=True)
            for t in tmp:
                values_to_add = {'filename': str(t), 'class': str(lab)}
                row_to_add = pd.Series(values_to_add, name=t)
                valid = valid.append(row_to_add, ignore_index=True)

        #tr = tr.apply(lambda x: pd.Series(trainList, columns=(x['filename'], x['class'])), axis=1)
        #valid = valid.apply(lambda x: pd.Series(validList, columns=(x['filename'], x['class'])), axis=1)

        print(f'\nDataFrame-train is empty: {tr.empty}')
        print(f'\nDataFrame-valid is empty: {valid.empty}')

        # Generators
        training_generator = datagen.flow_from_dataframe(dataframe=tr, target_size=(224, 224), batch_size=batchSize, class_mode="binary")
        validation_generator = datagen.flow_from_dataframe(dataframe=valid, target_size=(224, 224), batch_size=batchSize, class_mode="binary")

        base_model = keras.applications.VGG16(input_shape=(224, 224, 3), include_top = False, weights = None)

        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=True)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        # Unfreeze the base model
        model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

        # Train end-to-end. Be careful to stop before you overfit!
        historyTrain = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochNum)
        #historyTrain = model.fit(training_generator, epochs=epochNum, valid=validation_generator)

        hist_df = pd.DataFrame(historyTrain.history)
        hist_csv_file = resultFolderName + '\\vgg_historyTraining_' + name + '_' + str(m) + '.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        test_generator = datagen.flow_from_dataframe(test, target_size=(224, 224), batch_size=batchSize, class_mode="binary")

        loss, bin_acc, tp, tn, fp, fn = model.evaluate_generator(test_generator)
        recall = tp / (tp + fn + K.epsilon())
        precision = tp / (tp + fp + K.epsilon())
        f1 = (2 * recall * precision) / (recall + precision + K.epsilon())

        trainMetricsFile = resultFolderName + '\\vgg_metricOnTest_' + name + '_' + str(m) + '.txt'
        with open(trainMetricsFile, mode='w') as f:
            f.write('model ' + str(m) + ' on testset \n')
            f.write(str(bin_acc) + '\n')
            f.write(str(tp) + '\n')
            f.write(str(tn) + '\n')
            f.write(str(fp) + '\n')
            f.write(str(fn) + '\n')
            f.write(str(f1) + '\n')

        tst_acc.append(bin_acc)
        tst_f1.append(f1)

        if not (os.path.exists(resultFolderName + '\\vgg_ckpt' + str(m))):
            os.mkdir(resultFolderName + '\\vgg_ckpt' + str(m))

        model.save_weights(resultFolderName + '\\vgg_ckpt' + str(m) + '\\vgg_' + name + '_' + str(m))
        m = m + 1


    finalFile = resultFolderName + '\\vgg_finalmetrics_' + name + '.txt'
    with open(finalFile, mode='w') as f:
        f.write('mean tst_acc:' + '\n')
        f.write(str(np.mean(tst_acc)) + '\n')
        f.write('mean tst_f1:' + '\n')
        f.write(str(np.mean(tst_f1)) + '\n')
