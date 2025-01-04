import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super(FullyConvNetwork, self).__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        
        return output

# Define image pre-processing and post-processing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

def show_images(original, generated, title="Result"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(generated)
    plt.title("Generated")
    plt.axis("off")
    plt.suptitle(title)
    plt.show()

def main():
    # Model and paths
    model_path = "./checkpoints/pix2pix_0.pth"
    val_image_path = "./datasets/facades/val/86.jpg"
    train_image_path = "./datasets/facades/train/86.jpg"
    output_dir = "./results"

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = FullyConvNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Process validation image
    val_image = preprocess_image(val_image_path, device)
    with torch.no_grad():
        val_output = model(val_image)
    val_result = postprocess_image(val_output)

    # Process training image
    train_image = preprocess_image(train_image_path, device)
    with torch.no_grad():
        train_output = model(train_image)
    train_result = postprocess_image(train_output)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    val_result.save(os.path.join(output_dir, "val_result.jpg"))
    train_result.save(os.path.join(output_dir, "train_result.jpg"))
    print(f"Results saved to {output_dir} directory.")

    # Show results
    val_original = Image.open(val_image_path).convert("RGB")
    train_original = Image.open(train_image_path).convert("RGB")
    show_images(val_original, val_result, title="Validation Result")
    show_images(train_original, train_result, title="Training Result")



if __name__ == "__main__":
    main()
