import os
import torch
import torch.optim as optim
import torch.nn as nn
from models.classifier import ClassifierNet
from torch.utils.data import DataLoader
from dataset import create_data_loader, ReconstructionDataset
from visualizer import plot_training_loss
from models.decoder import YOLO_UNet
import util


class Trainer:
    def __init__(self, config, feature_extractor, num_epochs=120):
        self.config = config
        self.label_names = config["label_names"]
        self.feature_extractor = feature_extractor
        self.num_epochs = num_epochs

    def train_classifier(self):
        
        X, y = self.feature_extractor.extract_image_features_classification()
        train_loader = create_data_loader(X, y)

        _, channels = X.shape
        input_dim = channels      
        hidden_dim = 256     
        num_classes = len(self.label_names)

        classifier_model = ClassifierNet(input_dim, hidden_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)

        train_losses = []
        for epoch in range(self.num_epochs):
            classifier_model.train()
            running_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = classifier_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            print(f"Classifier Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

        plot_training_loss(train_losses, "Classifier")
        torch.save(classifier_model.state_dict(), "results/classifier_model.pth")
        print("Classifier model saved as 'classifier_model.pth'")

    def train_reconstruction(self):

        X, y = self.feature_extractor.extract_image_features_reconstruction()
        dataset = ReconstructionDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=1, collate_fn=util.custom_collate_fn)

        x_0, _ = next(iter(train_loader))[0]
        print("len(x_0)",x_0)
        x5_0 = x_0[5][0]
        x5_0_C, _,_ = x5_0.shape
        input_dim = x5_0_C 
        reconstruction_model = YOLO_UNet(input_dim)
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(reconstruction_model.parameters(), lr=0.001)

        train_losses = []
        for epoch in range(self.num_epochs):
            reconstruction_model.train()
            running_loss = 0.0
            
            for batch in train_loader:
                for batch_X, batch_y in batch:
                    optimizer.zero_grad()
                    outputs = reconstruction_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            print(f"Reconstruction Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

        plot_training_loss(train_losses, "Reconstruction")
        torch.save(reconstruction_model.state_dict(), "results/reconstruction_model.pth")
        print("Reconstruction model saved as 'reconstruction_model.pth'")

    def train(self):
        model_type = self.config.get("model_type")
        if model_type == "Classification":
            self.train_classifier()
        elif model_type == "Reconstruction":
            self.train_reconstruction()
        elif model_type == "Both":
            self.train_classifier()
            self.train_reconstruction()
        else:
            print("Enter the correct model_type. The options are 'Classification', 'Reconstruction' and 'Both'")