
Considerations:
	This code was programmed quickly due to the proposed timeout. Therefore, there are several things that can be improved and the organization can be changed.If it is possible to increase the timeframe for making improvements, I am willing to do them. And I am also available to do any other tests.


Organization:
	The requested are in the root.
	The codes are in Scripts directory. 
	The Bottleneck directory was used to store the decoding of the training images so that it was not necessary to decode them before each training. This directory is not need to analyze the images test.
	The TruthImages directory was used to store the true images and after it was compressed into a zip file in the Output directory.
	The inception_v3.pb was used to build the model. 
	The trained_model.pb is the trained model. 

Libraries:
	Tensorflow, cv2. 


Comands:
	To do retrain the model = python Scripts/Main.py retrain
	To rebuild bottleneck = python Scripts/Main.py rebuild_bottleneck
	To Create the ouputs from the test images:
		For all outputs = python Scripts/Image_Analyzer.py all
		For just the truth images = python Scripts/Image_Analyzer.py truth_images
		For just the truth predicts = python Scripts/Image_Analyzer.py truth_predicts
		For just the numpy file = python Scripts/Image_Analyzer.py numpy_file


 _______________________________________________________________________
|Thanks for the opportunity and I hope we can work and evolve together! |
|_______________________________________________________________________|
