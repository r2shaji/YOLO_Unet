import numpy as np
import torch, os, math
from PIL import Image
from torchvision import transforms

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # shape: (C, H, W)
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # Instead of (image_numpy + 1) / 2, if image is [0, 1] just multiply by 255.
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)
    else:
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

    
def pad_to_size( img, required_size=(256, 256)):

    if img.dim() != 4:
        raise ValueError("Expected img tensor of shape (B, C, H, W)")

    _, _, H, W = img.shape
    desired_width, desired_height = required_size

    pad_right = pad_bottom = 0
    final_width = max(desired_width, math.ceil(W / 32) * 32)
    final_height = max(desired_height, math.ceil(H / 32) * 32)

    pad_left = (final_width - W) // 2
    pad_right = final_width - W - pad_left

    pad_top = (final_height - H) // 2
    pad_bottom = final_height - H - pad_top

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_img = torch.nn.functional.pad(img, padding, mode='constant', value=0)
    return padded_img