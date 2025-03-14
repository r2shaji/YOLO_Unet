import torch
import torch.optim as optim
import torch.nn as nn
from models.classifier import ClassifierNet
from reconstruction import ReconstructionTrainer
import visualizer
from dataset import create_data_loader

class ClassifierTrainer:
    def __init__(self, config, feature_extractor, model_path = "", num_epochs=120, batch_size=32):

        self.config = config
        self.feature_extractor = feature_extractor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.label_names = config["label_names"]
        self.model_path = "results/classifier_model.pth"

        self.train_losses = []

        if model_path != "":
            self.model_path = model_path + "/classifier_model.pth"

        self.data_loader, X, y = self.create_train_loader()
        _, channels = X.shape
        print("channels",channels)
        input_dim = channels      
        hidden_dim = 256     
        num_classes = len(self.label_names)

        # Initialize the classifier model.
        self.classifier_model = ClassifierNet(input_dim, hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier_model.parameters(), lr=0.001)

    def create_train_loader(self):
        X, y = self.feature_extractor.extract_image_features_classification()
        return create_data_loader(X, y, self.batch_size), X, y

    def train(self):

        for epoch in range(self.num_epochs):
            self.classifier_model.train()
            running_loss = 0.0
            
            for batch_X, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.classifier_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            
            epoch_loss = running_loss / len(self.data_loader.dataset)
            self.train_losses.append(epoch_loss)
            print(f"Classifier Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
        

    def save_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        torch.save(self.classifier_model.state_dict(), self.model_path)
        print(f"Classifier model saved as '{self.model_path}'")

    def plot_loss(self):
        visualizer.plot_training_loss(self.train_losses, "Classifier", self.config["results_folder"])

    def load_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        self.classifier_model.load_state_dict(torch.load(self.model_path))
        print(f"Classifier model loaded from '{self.model_path}'")

        return self.classifier_model
    
    def test(self):
        self.load_model()

        all_preds = []
        all_labels = []

        self.classifier_model.eval()
        
        with torch.no_grad():
             for batch_X, batch_y in self.data_loader:
                outputs = self.classifier_model(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return all_labels, all_preds