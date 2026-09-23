import os
import lpips
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from torchvision import models, transforms
from torch.nn.functional import mse_loss

# Initialize LPIPS with AlexNet
loss_fn_alex = lpips.LPIPS(net='alex')

# Paths to the image folders
folder_img0 = "ablation/ours_ufuc"
folder_img1 = "ablation/tar_ufuc"

# List all files in the directories
files_img0 = sorted(os.listdir(folder_img0))
files_img1 = sorted(os.listdir(folder_img1))

# Ensure both folders have the same number of files
assert len(files_img0) == len(files_img1), "The folders do not contain the same number of images."

# Initialize lists to store the losses
lpips_losses = []
l1_losses = []
ssim_scores = []
fid_losses = []
rmse_scores = []

# Initialize a pre-trained Inception v3 model for FID calculation
inception_v3 = models.inception_v3(pretrained=True, transform_input=False).eval()

# Define the transform for Inception v3
transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_fid_features(img, model):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features

# Process each image pair
for file_img0, file_img1 in zip(files_img0, files_img1):
    # Load images
    img0 = Image.open(os.path.join(folder_img0, file_img0)).convert("RGB")
    img1 = Image.open(os.path.join(folder_img1, file_img1)).convert("RGB")

    # Resize images to the same size
    img0_resized = img0.resize((128, 128))
    img1_resized = img1.resize((128, 128))

    # Convert images to PyTorch tensors and normalize to [-1, 1]
    img0_tensor = (TF.to_tensor(img0_resized) - 0.5) * 2
    img1_tensor = (TF.to_tensor(img1_resized) - 0.5) * 2

    # Add batch dimension
    img0_tensor = img0_tensor.unsqueeze(0)
    img1_tensor = img1_tensor.unsqueeze(0)

    # Calculate LPIPS score
    lpips_score = loss_fn_alex(img0_tensor, img1_tensor)
    lpips_losses.append(lpips_score.item())

    # Calculate L1 loss
    l1_loss = torch.nn.functional.l1_loss(img0_tensor, img1_tensor)
    l1_losses.append(l1_loss.item())

    # Convert images back to [0, 1] range for SSIM calculation
    img0_norm = (img0_tensor.squeeze(0) + 1) / 2
    img1_norm = (img1_tensor.squeeze(0) + 1) / 2

    # Calculate SSIM score with specified win_size and data_range
    ssim_score, _ = ssim(img0_norm.numpy().transpose(1, 2, 0), img1_norm.numpy().transpose(1, 2, 0), 
                         full=True, multichannel=True, win_size=7, channel_axis=-1, data_range=1)
    ssim_scores.append(ssim_score)

    # Calculate FID features using the original PIL images
    fid_features0 = calculate_fid_features(img0, inception_v3)
    fid_features1 = calculate_fid_features(img1, inception_v3)
    fid_loss = mse_loss(fid_features0, fid_features1)
    fid_losses.append(fid_loss.item())

    # Calculate RMSE score
    mse_value = mean_squared_error(img0_norm.numpy(), img1_norm.numpy())
    rmse_value = mse_value ** 0.5
    rmse_scores.append(rmse_value)

# Calculate and print the average scores
average_lpips = sum(lpips_losses) / len(lpips_losses)
average_l1 = sum(l1_losses) / len(l1_losses)
average_ssim = sum(ssim_scores) / len(ssim_scores)
average_fid = sum(fid_losses) / len(fid_losses)
average_rmse = sum(rmse_scores) / len(rmse_scores)

print("Average LPIPS loss:", average_lpips)
print("Average L1 loss:", average_l1)
print("Average SSIM score:", average_ssim)
print("Average FID score:", average_fid)
print("Average RMSE score:", average_rmse)