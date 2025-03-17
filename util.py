import numpy as np
import torch, os
from PIL import Image
from torchvision import transforms

def tensor2im(input_image, imtype=np.uint8):

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_test_image(image, image_name, save_path="results"):

    image_numpy = tensor2im(image)
    os.makedirs(save_path, exist_ok=True)
    img_path = os.path.join(save_path, image_name)
    save_image(image_numpy, img_path)

def custom_collate_fn(batch):
    return batch

def get_image_tensor(image_path):

    image = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    image = to_tensor(image).unsqueeze(0)
    return image