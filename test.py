import torch
from models.classifier import ClassifierNet
from models.decoder import YOLO_UNet
from visualizer import plot_confusion_matrix
from torch.utils.data import DataLoader
from dataset import create_data_loader, ReconstructionDataset
import util


class Tester:
    def __init__(self, config, feature_extractor):
        self.config = config
        self.feature_extractor = feature_extractor
        self.label_names = config["label_names"]

        self.reconstruction_model = None
        self.model_classifier = None

    def evaluate_model_classifier(self):

        # Extract features for classification
        X, y = self.feature_extractor.extract_image_features_classification()
        test_loader = create_data_loader(X, y)

        _, channels = X.shape
        input_dim = channels      
        hidden_dim = 256     
        num_classes = len(self.label_names)
        self.model_classifier = ClassifierNet(input_dim, hidden_dim, num_classes)

        model_path = "results/classifier_model.pth"
        self.model_classifier.load_state_dict(torch.load(model_path))
        print(f"Classifier model loaded from '{model_path}'")

        all_preds = []
        all_labels = []

        self.model_classifier.eval()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model_classifier(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return all_labels, all_preds
                
    
    def test_classifier(self):
        
        all_labels, all_preds = self.evaluate_model_classifier()

        mapped_labels = list(map(lambda x: self.label_names.get(x), all_labels))
        mapped_preds = list(map(lambda x: self.label_names.get(x), all_preds))

        plot_confusion_matrix(mapped_labels, mapped_preds)
        print("Confusion matrix plotted.")

    def evaluate_model_reconstruction(self):

        # Extract features for reconstruction
        X, y = self.feature_extractor.extract_image_features_reconstruction()
        dataset = ReconstructionDataset(X, y)
        test_loader = DataLoader(dataset, batch_size=1, collate_fn=util.custom_collate_fn)

        x_0, _ = next(iter(test_loader))[0]
        x5_0 = x_0[5][0]
        print(x5_0.shape)
        x5_0_C, _,_ = x5_0.shape
        input_dim = x5_0_C   
        self.reconstruction_model = YOLO_UNet(input_dim)

        model_path = "results/reconstruction_model.pth"
        self.reconstruction_model.load_state_dict(torch.load(model_path))
        print(f"Classifier model loaded from '{model_path}'")

        self.reconstruction_model.eval()
        
        with torch.no_grad():
            count = 1
            for batch in test_loader:
                for batch_X, batch_y in batch:
                    output = self.reconstruction_model(batch_X)
                    util.save_to_image(output, "fake", count)
                    util.save_to_image(batch_y, "real", count)
                    count+=1


    def test_reconstruction(self):
        
        self.evaluate_model_reconstruction()

        print("Reconstructed images saved in 'results/plots' folder.")

    def test(self):
        model_type = self.config.get("model_type")
        if model_type == "Classification":
            self.test_classifier()
        elif model_type == "Reconstruction":
            self.test_reconstruction()
        elif model_type == "Both":
            self.test_classifier()
            self.test_reconstruction()
        else:
            print("Enter the correct model_type. The options are 'Classification', 'Reconstruction' and 'Both'")
