import torch
import skimage
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np

# 1) Transforms: keep as Tensor, then convert to numpy in HOG
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

celeba_train = datasets.CelebA(
    root="data",
    split="train",
    target_type="attr",
    transform=transform,
    download=False,  # True for first download
)

loader = DataLoader(celeba_train, batch_size=32, shuffle=True)

def hog_from_tensor(img_tensor):
    """
    img_tensor: torch.Size([3, H, W]) in [0,1]
    returns: 1D numpy array of HOG features
    """
    # C,H,W -> H,W,C and to numpy
    img = img_tensor.permute(1, 2, 0).numpy()
    gray = rgb2gray(img)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    return features

# Example: get one batch and compute HOG for first image
imgs, attrs = next(iter(loader))
hog_feat = hog_from_tensor(imgs[0])
print("HOG feature vector shape:", hog_feat.shape)
