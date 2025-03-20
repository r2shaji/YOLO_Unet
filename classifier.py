import torch
import torch.optim as optim
import torch.nn as nn
from models.classifier import ClassifierNet
import visualizer
import torch.utils.data
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset
import numpy as np

class ClassifierTrainer:
    def __init__(self, config, feature_extractor, model_path="", num_epochs=120, batch_size=32):
        self.config = config
        self.feature_extractor = feature_extractor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.label_names = config["label_names"]
        self.model_path = "results/classifier_model.pth"
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

        if model_path != "":
            self.model_path = model_path + "/classifier_model.pth"

        self.train_loader, self.val_loader, X, y = self.create_train_loader()
        input_dim = X.shape[1]
        hidden_dim = 256
        num_classes = len(self.label_names)
        self.classifier_model = ClassifierNet(input_dim, hidden_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.classifier_model.parameters(), lr=0.001)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        self.early_stop_patience = 8

    def create_train_loader(self):
        X, y = self.feature_extractor.extract_image_features_classification()
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, X, y

    def train(self):
        no_improvement_count = 0
        for epoch in range(self.num_epochs):
            self.classifier_model.train()
            running_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.classifier_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch_X.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)
            print(f"Classifier Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}")

            self.classifier_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.classifier_model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_running_loss += loss.item() * batch_X.size(0)
            val_loss = val_running_loss / len(self.val_loader.dataset)
            self.val_losses.append(val_loss)
            print(f"Classifier Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.classifier_model.state_dict()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {self.best_val_loss:.4f}")
                break
        
        if self.best_model_state:
            self.classifier_model.load_state_dict(self.best_model_state)

    def save_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        torch.save(self.classifier_model.state_dict(), self.model_path)
        print(f"Classifier model saved as '{self.model_path}'")

    def plot_loss(self):
        visualizer.plot_loss(self.train_losses, "Classification", "Training", self.config["results_folder"])
        visualizer.plot_loss(self.val_losses, "Classification", "Validation", self.config["results_folder"])

    def load_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        self.classifier_model.load_state_dict(torch.load(self.model_path))
        self.classifier_model.to(self.device)
        print(f"Classifier model loaded from '{self.model_path}'")
        return self.classifier_model

    def test(self):
        self.load_model()
        all_preds = []
        all_labels = []
        combined_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        self.classifier_model.eval()
        with torch.no_grad():
            for batch_X, batch_y in combined_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.classifier_model(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        return all_labels, all_preds
