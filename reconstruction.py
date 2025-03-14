import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ReconstructionDataset, transform_image
from models.decoder import YOLO_UNet
from models.networks import PerceptualLoss
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

        self.rec_criterion = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

        self.create_data_loader()
        input_dim = self.get_input_dim()
        self.reconstruction_model = YOLO_UNet(n_channels_in=input_dim)
        self.optimizer = optim.Adam(self.reconstruction_model.parameters(), lr=0.001)

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
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 collate_fn=util.custom_collate_fn)

    def get_input_dim(self):

        sample_batch, _, _ = next(iter(self.data_loader))[0] 
        x5_sample = sample_batch[5][0]
        input_dim = x5_sample.shape[0]

        return input_dim

    def train(self):

        for epoch in range(self.num_epochs):
            self.reconstruction_model.train()
            running_loss = 0.0
            for batch in self.data_loader:
                for features, target, _ in batch:
                    self.optimizer.zero_grad()
                    outputs = self.reconstruction_model(features)
                    loss = self.rec_criterion(outputs, target) + self.perceptual_loss(outputs, target)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                print("A batch done...")
            
            epoch_loss = running_loss / len(self.data_loader.dataset)
            self.train_losses.append(epoch_loss)
            print(f"Reconstruction Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

    
    def save_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        torch.save(self.reconstruction_model.state_dict(), self.model_path)
        print(f"Reconstruction model saved as '{self.model_path}'")

    def plot_loss(self):
        visualizer.plot_training_loss(self.train_losses, "Reconstruction", self.config["results_folder"])

    def load_model(self, model_path=""):
        if model_path != "":
            self.model_path = model_path
        self.reconstruction_model.load_state_dict(torch.load(self.model_path))
        print(f"Reconstruction model loaded from '{self.model_path}'")

        return self.reconstruction_model
    
    def test(self):
        self.load_model()
        self.reconstruction_model.eval()

        with torch.no_grad():
            count = 1
            for batch in self.data_loader:
                for batch_X, batch_y in batch:
                    output = self.reconstruction_model(batch_X)
                    util.save_test_image(output, "fake", count, self.config["results_folder"])
                    util.save_test_image(batch_y, "real", count, self.config["results_folder"])
                    count+=1



