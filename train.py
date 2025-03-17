import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from reconstruction import ReconstructionTrainer
from classifier import ClassifierTrainer
from dataset import load_label
from models.networks import PerceptualLoss
import visualizer, util

class Trainer:
    def __init__(self, config, feature_extractor, num_epochs=120):
        self.config = config
        self.label_names = config["label_names"]
        self.feature_extractor = feature_extractor
        self.num_epochs = num_epochs

    def train_classifier(self):

        class_trainer = ClassifierTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)
        class_trainer.train()
        class_trainer.plot_loss()
        class_trainer.save_model()

    def train_reconstruction(self):

        recon_trainer = ReconstructionTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)
        recon_trainer.train()
        recon_trainer.plot_loss()
        recon_trainer.save_model()

    def train_both(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        recon_trainer = ReconstructionTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)
        train_loader = recon_trainer.data_loader
        reconstruction_model = recon_trainer.load_model().to(device)

        class_trainer = ClassifierTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)
        classifier_model = class_trainer.load_model().to(device)

        rec_criterion = nn.MSELoss().to(device)
        rec_perceptualLoss = PerceptualLoss().to(device)
        class_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = optim.Adam(list(reconstruction_model.parameters()) + list(classifier_model.parameters()), lr=0.001)

        train_losses = []
        for epoch in range(self.num_epochs):
            reconstruction_model.train()
            classifier_model.train()
            running_loss = 0.0

            for batch in train_loader:
                for features, real_im, sharp_path in batch:
                    optimizer.zero_grad()
                    features = [f.to(device) for f in features]
                    real_im = real_im.to(device)

                    rec_output = reconstruction_model(features)
                    rec_loss = rec_criterion(rec_output, real_im) + rec_perceptualLoss(rec_output, real_im)

                    gt = load_label(sharp_path, self.config["label_folder"])
                    cropped_char_features, true_labels = self.feature_extractor.extract_image_features(rec_output, gt)
                    cropped_char_features = torch.stack(cropped_char_features).to(device)
                    true_labels = torch.tensor(true_labels).to(device)
                    class_outputs = classifier_model(cropped_char_features)
                    class_loss = class_criterion(class_outputs, true_labels)

                    total_loss = rec_loss + class_loss
                    total_loss.backward()
                    optimizer.step()
                    running_loss += total_loss.item()

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            print(f"Reconstruction + Classification Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

        recon_trainer.save_model()
        class_trainer.save_model()

        visualizer.plot_training_loss(train_losses, "Reconstruction + Classification", self.config["results_folder"])

    def train(self):

        model_type = self.config.get("model_type")
        if model_type == "Classification":
            self.train_classifier()
        elif model_type == "Reconstruction":
            self.train_reconstruction()
        elif model_type == "Both":
            self.train_both()
        else:
            print("Enter the correct model_type. The options are 'Classification', 'Reconstruction' and 'Both'")
