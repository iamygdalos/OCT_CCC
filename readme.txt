#1. Install Anaconda3, create virtual environment and install following packages:

conda create --name name
conda install python=3.6
conda install -c anaconda keras-gpu
conda install -c conda-forge opencv
conda install matplotlib
conda install scikit-learn
conda install -c anaconda pandas
conda install -c anaconda pillow


#2. Open Project
(Folders readoct, imageprocessing, movingFiles, training containing *.py Files)

readoct
	2 python files: read in .oct files from folder, extract images in X/Y, X/Z, Y/Z and save to other folder
		using A-Scan information for intensity scaling
		using histogram information for intensity scaling
		
	ReadFileListAndExtractOct(fileList, folder, folder_export) 
		fileList - patientcodes that should be extracted in "patient1.oct;\npatient2.oct\n...."
		folder - basefolder where .oct files are stored
		folder_export - where extracted data should be stored (does not have to exist, will be created)
			Extracted Data will be stored in folder_export/patientcode/orientationNum
		
FROM NOW ON FILES HAVE TO BE IN DIRECTORY STRUCTURE: startfolder/diagnose/patientcode/orientationNum

imageprocessing
	Perform image preprocessing on extracted images from .oct files (specify orientiation with 'imgNum' variable!)
movingFiles
	CopySingleFiles
		moves files from folder into other folder (usage for example for taking only every second, every fifth..., parameter in CreateDataset method)
	Merge1And2
		merge files from two folders (for example two orientations) into one
training
	train_AllDatasets - create cross validation splits for data and calls other files for performing training with different models
		what you need: 
			file1: text file with patientcodes and diagnose for training (train and valid) in "patientcode;diagnose;\npatientcode;diagnose;\n..."
			file2: text file with patientcodes and diagnose for test "patientcode;diagnose;\npatientcode;diagnose;\n..."
		
		
		TrainOnModel (keras_train_VGG16_AllData & keras_train_Xcpetion_AllData)
			Training(imgPath, skfSplits, availableLabels, patients, labels, patientsTest, labelsTest, resultsFolderName, name, batchSize, epochNum):
				imgPath - path to dataset used for training (should contain specified patientcodes)
				skfSplits - array of arrays defining the number of Cross validation splits (CVRUNS) and train/valid elements (CV split)
				availableLabels - available labels/diagnoses (have to match directory names and labelnames given in file1 & file2!!)
				patients - array of patientcodes (read from file1)
				labels - array of labels for patientcodes from 'patients' (read from file1)
				patientsTest - array of patientcodes used as test (read from file2)
				labelsTest - array of labels for patientcodes from 'patientsTest' (read from file2)
				resultsFolderName - where to save model results (training results etc.)
				name - name for saving model
				batchSize
				epochNum
	
		
				procedure method (for CVRUNS):					
				Creates Datagenerators for test, validation and train (based on current CV split)
					--> DataGenerators are used during training to load only required data for current batch
					--> take in path to every single image 
			
				Creates basemodel (pretrained) and does modifications (e.g. last layer for binary output)
				Freezes basemodel, trains for one epoch only last layer (finetuning) 
				Unfreezes basemodel, trains whole model for specified number of epochs
				Saves training information (accuracy,FP,FN,... etc. on every epoch) with specified name 
				Uses model for predictions on Testset and save test information (accuracy,FP,FN,... etc.)
				
		Outputs: 
			in resultsFolderName 4 types of files are created with the name given for the model for training 
			1. xception_historyBegin_name_CVRUN.csv
				training results during first run from finetuning when all layers, except last, are frozen for each cv run
			2. xception_historyTraining_name_CVRUN.csv
				training results during all other epochs from finetuning when all layers are unfrozen for each cv run
			3. xception_metricOnTest_name_CVRUN.csv
				results of final model on test set for each cv run
			4. xception_finalmetrics_name
				mean accurarcy and f1-score on testset over all cv runs
				
		LoadAndPredictWithTrainedNetworks
			Functions for loading Xcpetion or VGG and predict on images, B-Scan and C-Scan and calculate runtime
			FeatureMapsPrinting
plotResults
	functions for result (F1 & Loss) plotting.. As Input folder always use 'resultsFolderName' 	
	CalcMetricsForBAndCScanPredictions - calcs FP, FN, TP, TN, Acc etc. for B- and C-Scan (from LoadAndPredictWithTrainedNetworks outputs)
	



