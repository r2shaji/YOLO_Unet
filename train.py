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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_trainer = ClassifierTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)
        self.recon_trainer = ReconstructionTrainer(self.config, self.feature_extractor, self.config["results_folder"], num_epochs=self.num_epochs, batch_size=4)

        self.rec_criterion = nn.MSELoss().to(self.device)
        self.rec_perceptualLoss = PerceptualLoss().to(self.device)
        self.class_criterion = nn.CrossEntropyLoss().to(self.device)
        

    def _compute_loss(self, features, real_im, sharp_path, reconstruction_model, classifier_model):
        features = [f.to(self.device) for f in features]
        real_im = real_im.to(self.device)
        rec_output = reconstruction_model(features)
        rec_loss = self.rec_criterion(rec_output, real_im) + self.rec_perceptualLoss(rec_output, real_im)
        gt = load_label(sharp_path, self.config["label_folder"])
        cropped_char_features, true_labels = self.feature_extractor.extract_image_features(rec_output, gt)
        cropped_char_features = torch.stack(cropped_char_features).to(self.device)
        true_labels = torch.tensor(true_labels).to(self.device)
        class_outputs = classifier_model(cropped_char_features)
        class_loss = self.class_criterion(class_outputs, true_labels)
        total_loss = rec_loss + class_loss
        return total_loss

    def train_classifier(self):

        self.class_trainer.train()
        self.class_trainer.plot_loss()
        self.class_trainer.save_model()

    def train_reconstruction(self):

        self.recon_trainer.train()
        self.recon_trainer.plot_loss()
        self.recon_trainer.save_model()

    def train_both(self):
        

        train_loader = self.recon_trainer.train_loader
        val_loader = self.recon_trainer.val_loader
        
        reconstruction_model = self.recon_trainer.load_model().to(self.device)
        classifier_model = self.class_trainer.load_model().to(self.device)

        optimizer = optim.Adam(list(reconstruction_model.parameters()) + list(classifier_model.parameters()), lr=0.001)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_recon_state = None
        best_classifier_state = None
        
        #Training
        for epoch in range(self.num_epochs):
            reconstruction_model.train()
            classifier_model.train()
            running_loss = 0.0
            for batch in train_loader:
                for features, real_im, sharp_path in batch:
                    
                    total_loss = self._compute_loss(features, real_im, sharp_path, reconstruction_model, classifier_model)
                    total_loss.backward()
                    optimizer.step()
                    running_loss += total_loss.item()
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            print(f"Reconstruction + Classification Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}")
            
            # Validation
            reconstruction_model.eval()
            classifier_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    for features, real_im, sharp_path in batch:
                        total_loss = self._compute_loss(features, real_im, sharp_path, reconstruction_model, classifier_model)
                        val_running_loss += total_loss.item()
            val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Reconstruction + Classification Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}")
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_recon_state = reconstruction_model.state_dict()
                best_classifier_state = classifier_model.state_dict()

        # Save the best models
        if not best_classifier_state:
            self.recon_trainer.save_model()
            self.class_trainer.save_model()
            print("Validation loss didn't decrease. Saving the final model")
        else:
            reconstruction_model.load_state_dict(best_recon_state)
            classifier_model.load_state_dict(best_classifier_state)
            self.recon_trainer.save_model()
            self.class_trainer.save_model()
            print(f"Models saved with validation loss: {best_val_loss:.4f}")
        
        visualizer.plot_loss(train_losses, "Reconstruction + Classification", "Training", self.config["results_folder"])
        visualizer.plot_loss(val_losses, "Reconstruction + Classification", "Validation", self.config["results_folder"])


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
