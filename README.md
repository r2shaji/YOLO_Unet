# YOLO_Unet

To train run the command:
'''
python main.py --mode train --config "config json path"
'''

To test run the command:
'''
python main.py --mode test --config "config json path"
'''

The config file contains the following,
"sharp_image_folder": "folder path to sharp images",
"blur_image_folder": "folder path to blur versions of the sharp. Should be of same name",
"label_folder":"folder path to labels of the images. Should be of same name. Labels in YOLO format",
"yolo_path":"Path to the YOLO model",
"model_type": "The type of model to be trained or tested. Options: [Reconstruction, Classification, Both]" ,
"results_folder": "Path to store the results including plots, the trained model, reconstructed images, etc",
"num_epochs":number of epochs,
"label_names": Example format: {"10": "a", "11": "b", "12": "c"}
