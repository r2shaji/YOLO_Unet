A framework for training and testing YOLO with added reconstruction and classification head.


## Usage
### Training
```
python main.py --mode train --config "path/to/config.json"
```
#### For reconstruction head training

Inputs the sharp and blur image folders to train the reconstruction model. Also takes in the YOLO model to extract the layer embeddings. Outputs the trained model, training and validation plots.

#### For classification head training

Inputs the sharp image and label folders to train the classification model. Also takes in the YOLO model to extract the layer embeddings. Outputs the trained model, training and validation plots.

#### For both training

Inputs the sharp image, blur image and label folders to train the classification model. Also takes in the YOLO model to extract the layer embeddings. Outputs the trained model, training and validation plots. Also update the weights of the YOLO model trained in this multi task setup.

### Testing
```
python main.py --mode test --config "path/to/config.json"
```

#### For reconstruction head testing

Inputs the blur image folder. Outputs the reconstructed images. 

#### For classification head testing

Inputs the sharp image folder. Outputs the confusion matrix chart and prints the score.

### Configuration File

The configuration JSON file should contain the following fields:
```
{
  "sharp_image_folder": "path/to/sharp/images",
  "blur_image_folder": "path/to/blur/images", 
  "label_folder": "path/to/labels", 
  "yolo_path": "path/to/yolo/model",
  "model_type": "Reconstruction | Classification | Both",
  "results_folder": "path/to/save/results",
  "num_epochs": 50,
  "label_names": {
    "10": "a",
    "11": "b",
    "12": "c"
  }
}
```

### Field Descriptions

sharp_image_folder → Folder containing sharp/original images.

blur_image_folder → Folder containing blurred versions of the sharp images.

Note: File names must match those in sharp_image_folder.

label_folder → Folder containing labels in YOLO format.

Note: File names must match the corresponding images.

yolo_path → Path to the YOLO model weights.

model_type → Type of task:

Reconstruction → Train or test the reconstruction head.

Classification → Train or test the classification head.

Both → Train or test both in multi-task setup.

results_folder → Path to store results (trained models, plots, reconstructed images, etc).

num_epochs → Number of training epochs.

label_names → Mapping of class IDs to class names.

### Example
```
{
  "sharp_image_folder": "./data/sharp",
  "blur_image_folder": "./data/blur",
  "label_folder": "./data/labels",
  "yolo_path": "./weights/yolov5s.pt",
  "model_type": "Both",
  "results_folder": "./results",
  "num_epochs": 100,
  "label_names": {
    "1": "a",
    "1": "b",
    "1": "c"
  }
}
```

### Link to the dataset and the YOLO models trained in single task and multi task setup

https://drive.google.com/file/d/1qbpJ51lSAAx0q-qnvCzaY3ubHi8swUzK/view?usp=sharing

Notes

Ensure that sharp, blur, and label folders have files with the same names.

Labels must follow YOLO format.

The model can be run in different modes (Reconstruction, Classification, Both) depending on the config.
