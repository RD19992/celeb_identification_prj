import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1) Basic image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 2) Load CelebA
celeba_train = datasets.CelebA(
    root="data",              # folder where it will be stored
    split="train",            # "train" | "valid" | "test" | "all"
    target_type="attr",       # or "identity", "bbox", "landmarks"
    transform=transform,
    download=False,            # set False if already downloaded
)

loader = DataLoader(celeba_train, batch_size=64, shuffle=True)

# Example: get one batch
imgs, attrs = next(iter(loader))
print(imgs.shape, attrs.shape)  # (64, 3, 128, 128), (64, 40)
