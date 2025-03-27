import torch
import os
import torch.optim as optim
from torch.utils.data import ConcatDataset
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import ReconstructionDataset, transform_image
from models.decoder import YOLO_UNet
from models.networks import PerceptualLoss, SSIMLoss
import visualizer, util

class ReconstructionTrainer:
    def __init__(self, config, feature_extractor, model_path="", num_epochs=120, batch_size=16):
        self.config = config
        self.feature_extractor = feature_extractor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_path = "results/reconstruction_model.pth"
        if model_path != "":
            self.model_path = model_path + "/reconstruction_model.pth"
        self.train_losses = []
        self.val_losses = []

        self.train_loss_L2 = []
        self.train_loss_L1 = []
        self.train_loss_perceptual = []
        self.val_loss_L2 = []
        self.val_loss_L1 = []
        self.val_loss_perceptual = []

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.rec_criterion = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.create_data_loader()
        input_dim = self.get_input_dim()
        self.reconstruction_model = YOLO_UNet(n_channels_in=input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_model.to(self.device)
        self.optimizer = optim.Adam(self.reconstruction_model.parameters(), lr=0.003)

        self.ssim_loss = SSIMLoss(window_size=11, size_average=True, val_range=1)
        self.rec_criterion_L1 = nn.L1Loss()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        self.early_stop_patience = 10

    def create_data_loader(self):
        to_tensor = transforms.ToTensor()
        dataset = ReconstructionDataset(
            blur_image_paths=self.feature_extractor.blur_image_paths,
            sharp_image_folder=self.feature_extractor.sharp_image_folder,
            model=self.feature_extractor.model,
            embed_layers=self.feature_extractor.embed_layers,
            transform_image_func=transform_image,
            to_tensor=to_tensor
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=util.custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=util.custom_collate_fn)

    def get_input_dim(self):
        sample_batch, _, _ = next(iter(self.train_loader))[0]
        x5_sample = sample_batch[5][0]
        input_dim = x5_sample.shape[0]
        return input_dim

    def train(self):
        no_improvement_count = 0
        for epoch in range(self.num_epochs):
            self.reconstruction_model.train()
            running_loss = 0.0
            loss_L2 = 0.0
            loss_L1 = 0.0
            loss_perceptual = 0.0
            for batch in self.train_loader:
                for features, target, _ in batch:
                    self.optimizer.zero_grad()
                    features = [f.to(self.device) for f in features]
                    target = target.to(self.device)
                    outputs = self.reconstruction_model(features)
                    L2loss = self.rec_criterion(outputs, target)
                    perceptual_loss = self.perceptual_loss(outputs, target)
                    L1loss = self.rec_criterion_L1(outputs, target)
                    # perceptual_loss = self.ssim_loss(outputs, target)
                    loss =  10 * L2loss + 0.1* perceptual_loss + 20 * L1loss
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                    loss_L2 += L2loss.item()
                    loss_L1 += L1loss.item()
                    loss_perceptual += perceptual_loss.item()
                print("A batch done...")
            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)
            print("self.train_losses",self.train_losses)

            self.train_loss_perceptual.append(loss_perceptual/len(self.train_loader.dataset))
            self.train_loss_L2.append(loss_L2/len(self.train_loader.dataset))
            self.train_loss_L1.append(loss_L1/len(self.train_loader.dataset))

            print(f"Reconstruction Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}")

            self.reconstruction_model.eval()
            val_running_loss = 0.0
            val_loss_L2 = 0.0
            val_loss_L1 = 0.0
            val_loss_perceptual = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    for features, target, _ in batch:
                        features = [f.to(self.device) for f in features]
                        target = target.to(self.device)
                        outputs = self.reconstruction_model(features)
                        val_L2loss = self.rec_criterion(outputs, target)
                        perceptual_loss = self.perceptual_loss(outputs, target)
                        val_L1loss = self.rec_criterion_L1(outputs, target)
                        # perceptual_loss = self.ssim_loss(outputs, target)
                        val_loss = 10 * val_L2loss +  0.1* perceptual_loss + 20 * val_L1loss
                        val_running_loss += val_loss.item()

                        val_loss_L2 += val_L2loss.item()
                        val_loss_L1 += val_L1loss.item()
                        val_loss_perceptual += perceptual_loss.item()
            val_loss = val_running_loss / len(self.val_loader.dataset)
            self.val_losses.append(val_loss)

            self.val_loss_perceptual.append(loss_perceptual/len(self.val_loader.dataset))
            self.val_loss_L2.append(val_loss_L2/len(self.val_loader.dataset))
            self.val_loss_L1.append(val_loss_L1/len(self.val_loader.dataset))

            print(f"Reconstruction Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.reconstruction_model.state_dict()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {self.best_val_loss:.4f}")
                break

        if self.best_model_state:
            self.reconstruction_model.load_state_dict(self.best_model_state)

    def save_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        torch.save(self.reconstruction_model.state_dict(), self.model_path)
        print(f"Reconstruction model saved as '{self.model_path}'")

    def plot_loss(self):
        visualizer.plot_loss(self.train_losses, "Reconstruction", "Training", self.config["results_folder"])
        visualizer.plot_loss(self.val_losses, "Reconstruction", "Validation", self.config["results_folder"])
        visualizer.plot_loss(self.train_loss_L2, "Reconstruction", "Training L2", self.config["results_folder"])
        visualizer.plot_loss(self.train_loss_L1, "Reconstruction", "Training L1", self.config["results_folder"])
        visualizer.plot_loss(self.train_loss_perceptual, "Reconstruction", "Training Perceptual", self.config["results_folder"])
        visualizer.plot_loss(self.val_loss_L2, "Reconstruction", "Validation L2", self.config["results_folder"])
        visualizer.plot_loss(self.val_loss_L1, "Reconstruction", "Validation L1", self.config["results_folder"])
        visualizer.plot_loss(self.val_loss_perceptual, "Reconstruction", "Validation Perceptual", self.config["results_folder"])

    def load_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        self.reconstruction_model.load_state_dict(torch.load(self.model_path))
        self.reconstruction_model.to(self.device)
        print(f"Reconstruction model loaded from '{self.model_path}'")
        return self.reconstruction_model

    def test(self):
        self.load_model()
        self.reconstruction_model.eval()
        combined_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=util.custom_collate_fn)
        with torch.no_grad():
            for batch in combined_loader:
                for batch_X, batch_y, image_path in batch:
                    image_name = os.path.basename(image_path)
                    batch_X = [x.to(self.device) for x in batch_X]
                    batch_y = batch_y.to(self.device)
                    output = self.reconstruction_model(batch_X)
                    util.save_test_image(output, image_name, self.config["results_folder"] + "/fake")
                    util.save_test_image(batch_y, image_name, self.config["results_folder"] + "/real")

