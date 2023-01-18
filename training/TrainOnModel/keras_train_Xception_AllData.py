import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

def GetImageFiles(dirPath):
    return list(Path(dirPath).rglob('*.[pP][nN][gG]'))

def Training(imgPath, skfSplits, availableLabels, patients, labels, patientsTest, labelsTest, resultsFolderName, name, batchSize, epochNum, tumortype):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    if not (os.path.exists(resultsFolderName)):
        os.mkdir(resultsFolderName)

    datagen = ImageDataGenerator(rescale=1 / 255)

    test = pd.DataFrame(columns=('filename', 'class'))

    i = 0
    for pat in patientsTest:
        lab = labelsTest[i]
        i = i + 1
        patPath = imgPath
        if lab == "T":
            patPath = patPath + tumortype + "\\"
        else:
            patPath = patPath + "Parenchym\\"

        patPath = patPath + pat + "\\" + str(2) + "\\"
        tmp = GetImageFiles(patPath)
        print('loading patient..' + pat)
        for t in tmp:
            values_to_add = {'filename': str(t), 'class': str(lab)}
            row_to_add = pd.Series(values_to_add, name=t)
            test = test.append(row_to_add)

    m = 0
    tst_acc = []
    tst_f1 = []
    for s in skfSplits:
        train = s[0]
        testIdx = s[1]

        tr = pd.DataFrame(columns=('filename', 'class'))
        valid = pd.DataFrame(columns=('filename', 'class'))

        for i in train:
            pat = patients[i]
            lab = labels[i]
            patPath = imgPath

            if lab == "T":
                patPath = patPath + tumortype + "\\"
            else:
                patPath = patPath + "Parenchym\\"
            print('loading patient..' + pat)
            patPath = patPath + pat + "\\" + str(2) + "\\"
            tmp = GetImageFiles(patPath)
            for t in tmp:
                values_to_add = {'filename': str(t), 'class': str(lab)}
                row_to_add = pd.Series(values_to_add, name=t)
                tr = tr.append(row_to_add)

        for i in testIdx:
            pat = patients[i]
            lab = labels[i]

            patPath = imgPath
            if lab == "T":
                patPath = patPath + tumortype + "\\"
            else:
                patPath = patPath + "Parenchym\\"
            print('loading patient..' + pat)
            patPath = patPath + pat  + "\\" + str(2) + "\\"
            tmp = GetImageFiles(patPath)
            for t in tmp:
                values_to_add = {'filename': str(t), 'class': str(lab)}
                row_to_add = pd.Series(values_to_add, name=t)
                valid = valid.append(row_to_add)

        # Generators
        training_generator = datagen.flow_from_dataframe(tr, x_col="filename", y_col="class", target_size=(299, 299), batch_size=batchSize, class_mode="binary")
        validation_generator = datagen.flow_from_dataframe(valid, x_col="filename", y_col="class", target_size=(299, 299), batch_size=batchSize, class_mode="binary")

        # create base model
        base_model = keras.applications.Xception(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(299, 299, 3),
            include_top=False)  # Do not include the ImageNet classifier at the top.

        base_model.trainable = False

        # do modifications
        inputs = keras.Input(shape=(299, 299, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        #compile:)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

        #fine tuning first step: train with base model parameters frozen --> train only last layer
        historyFirst = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1)

        # Unfreeze the base model
        base_model.trainable = True
        model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

        # Train whole model


        historyTrain = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochNum)

        ##training finished:) now save results from training

        hist_dfF = pd.DataFrame(historyFirst.history)
        hist_csv_fileF = resultsFolderName + '\\xception_historyBegin_' + name + '_' + str(m) + '.csv'
        with open(hist_csv_fileF, mode='w') as f:
            hist_dfF.to_csv(f)

        hist_df = pd.DataFrame(historyTrain.history)
        hist_csv_file = resultsFolderName + '\\xception_historyTraining_' + name + '_' + str(m) + '.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        #test model on test dataset and save results
        test_generator = datagen.flow_from_dataframe(test, x_col="filename", y_col="class", target_size=(299, 299), batch_size=batchSize, class_mode="binary")

        loss, bin_acc, tp, tn, fp, fn = model.evaluate_generator(generator=test_generator)
        recall = tp / (tp + fn + K.epsilon())
        precision = tp / (tp + fp + K.epsilon())
        f1 = (2 * recall * precision) / (recall + precision + K.epsilon())

        trainMetricsFile = resultsFolderName + '\\xception_metricOnTest_' + name + '_' + str(m) + '.txt'
        with open(trainMetricsFile, mode='w') as f:
            f.write('Run ' + str(m) + ' on testset \n')
            f.write('Accuarcy: ' + str(bin_acc) + '\n')
            f.write('TP: ' + str(tp) + '\n')
            f.write('TN: ' + str(tn) + '\n')
            f.write('FP: ' + str(fp) + '\n')
            f.write('FN: ' + str(fn) + '\n')
            f.write('F1: ' + str(f1) + '\n')

        tst_acc.append(bin_acc)
        tst_f1.append(f1)

        if not (os.path.exists(resultsFolderName + '\\xception_ckpt' + str(m))):
            os.mkdir(resultsFolderName + '\\xception_ckpt' + str(m))


        #save model trained weights
        model.save_weights(resultsFolderName + '\\xception_ckpt' + str(m) + '\\xception_' + name + '_' + str(m))
        m = m + 1

        keras.backend.clear_session()

    finalFile = resultsFolderName + '\\xception_finalmetrics_' + name + '.txt'
    with open(finalFile, mode='w') as f:
        f.write('mean tst_acc:' + '\n')
        f.write(str(np.mean(tst_acc)) + '\n')
        f.write('mean tst_f1:' + '\n')
        f.write(str(np.mean(tst_f1)) + '\n')

