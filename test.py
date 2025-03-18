import glob, os
from classifier import ClassifierTrainer
from reconstruction import ReconstructionTrainer
from visualizer import plot_confusion_matrix


class Tester:
    def __init__(self, config, feature_extractor):
        self.config = config
        self.feature_extractor = feature_extractor
        self.label_names = config["label_names"]   
    
    def test_classifier(self):
        
        class_trainer = ClassifierTrainer(self.config, self.feature_extractor, model_path = self.config["results_folder"], num_epochs=120, batch_size=4)
        all_labels, all_preds = class_trainer.test()
        mapped_labels = list(map(lambda x: self.label_names.get(x), all_labels))
        mapped_preds = list(map(lambda x: self.label_names.get(x), all_preds))

        plot_confusion_matrix(mapped_labels, mapped_preds, self.config["results_folder"])
        print("Confusion matrix plotted.")

    def test_reconstruction(self):
        
        recon_trainer = ReconstructionTrainer(self.config, self.feature_extractor,  model_path = self.config["results_folder"], num_epochs=120, batch_size=4)
        recon_trainer.test()
        print(f"Reconstructed images saved in '{self.config["results_folder"]+"/fake"}' folder.")

    def test_both(self):
        recon_trainer = ReconstructionTrainer(self.config, self.feature_extractor,  model_path = self.config["results_folder"], num_epochs=120, batch_size=4)
        recon_trainer.test()

        self.feature_extractor.sharp_image_paths = sorted(
                        glob.glob(os.path.join(self.config["results_folder"]+"/fake", "*.jpg")) 
                        )

        class_trainer = ClassifierTrainer(self.config, self.feature_extractor,  model_path = self.config["results_folder"], num_epochs=120, batch_size=4)
        all_labels, all_preds = class_trainer.test()
        mapped_labels = list(map(lambda x: self.label_names.get(x), all_labels))
        mapped_preds = list(map(lambda x: self.label_names.get(x), all_preds))

        plot_confusion_matrix(mapped_labels, mapped_preds,self.config["results_folder"])
        print("Confusion matrix plotted.")


    def test(self):
        model_type = self.config.get("model_type")
        if model_type == "Classification":
            self.test_classifier()
        elif model_type == "Reconstruction":
            self.test_reconstruction()
        elif model_type == "Both":
            self.test_both()
        else:
            print("Enter the correct model_type. The options are 'Classification', 'Reconstruction' and 'Both'")
