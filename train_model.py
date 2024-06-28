import pickle
from typing import Any, Mapping
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import torch
import torch.nn as nn

from danet import DANet
from model import LFNet
from pymlab.train import TrainResults
from pymlab.utils import make_file


# custom End-Point Error (EPE) Loss class
class EPE_LOSS(nn.Module):
    def __init__(self):
        super(EPE_LOSS, self).__init__()

    def forward(self, output, mask):
        # Calculate EPE loss
        epe_loss = torch.mean(torch.sqrt(torch.sum((output - mask) ** 2, dim=1)))
        return epe_loss


# Dataset class to handle loading and transforming images and masks
class CustomDataset(Dataset):
    def __init__(self, images_dir, transform=None, is_label=False):
        self.images_dir = images_dir
        self.images_list = sorted(os.listdir(images_dir))
        self.transform = transform
        self.is_label = is_label

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images_list[idx])
        image = Image.open(img_name)
        if self.is_label:
            image = image.convert("L")  # Convert labels to grayscale
        if not self.is_label:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

# Function to get all file paths with specific extensions from a directory
def get_file_paths(directory, extensions):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
    return paths

# Function to train the model
async def train_model(
    dataset_path: str,
    parameters: dict[str, Any],
    result_id: str,
    **kwargs
) -> TrainResults:

    # Retrieve training parameters
    image_size = parameters.get('image_size', 128)
    epochs = parameters.get('epochs', 2)
    learning_rate = parameters.get('learning_rate', 0.0001)
    b1 = parameters.get('b1', 0.5)
    b2 = parameters.get('b2', 0.99)
    
    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Directories for images and masks
    image_dir = dataset_path + "/train"
    mask_dir = dataset_path + "/train_mask"

    # # Get file paths for images and masks
    # image_paths = get_file_paths(image_dir, ['jpg'])
    # mask_paths = get_file_paths(mask_dir, ['png'])
    #
    # # Ensure datasets are sorted
    # image_paths = sorted(image_paths)
    # mask_paths = sorted(mask_paths)
    # print('images:', image_paths)
    # print('masks:', mask_paths)

    # Create dataset and dataloader
    dataset_image = CustomDataset(image_dir, transform=transform)
    dataset_mask = CustomDataset(mask_dir, transform=transform, is_label=True)
    # dataset_image = CustomDataset(image_dir, transform=transform)

    dataloader_image = DataLoader(dataset_image, batch_size=10, shuffle=False)
    dataloader_mask = DataLoader(dataset_mask, batch_size=10, shuffle=False)

    # Initialize model_lfnet, loss_lf functions, and optimizer_lfnet
    nclass = 2
    model_lfnet = LFNet()
    model_danet = DANet(nclass=nclass)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lfnet.to(device)
    model_danet.to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    epe_loss_fn = EPE_LOSS()

    optimizer_lfnet = torch.optim.Adam(model_lfnet.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_danet = torch.optim.Adam(model_danet.parameters(), lr=learning_rate, betas=(b1, b2))


    # Path for logging training progress
    logger_path = make_file(result_id, 'train.log')

    all_da = []
    all_lf = []

    # Training loop
    for epoch in range(epochs):
        model_lfnet.train()
        model_danet.train()
        for image, mask in zip(dataloader_image, dataloader_mask):
            images = image.to(device)
            masks = mask.to(device)

            optimizer_lfnet.zero_grad()
            optimizer_danet.zero_grad()

            outputs_lf = model_lfnet(images)
            print('lfnet output generated')
            # outputs_lf = outputs_lf.convert("L")
            print('outputs_lf:', outputs_lf)

            orig_label = (images, masks)  # RGB + label
            print('orig_label done')
            orig_pred = (images, outputs_lf)  # RGB + pred
            print('orig_pred done')
            label_pred = (masks, outputs_lf)  # label + pred
            print('label_pred done')

            # Forward pass through DANet
            outputs_da = model_danet(orig_label, orig_pred, label_pred)
            print('danet output generated')

            # Get predictions
            preds = torch.argmax(outputs_da, dim=1).cpu().numpy().flatten()
            all_da.extend(preds)
            all_lf.extend(outputs_lf.cpu().numpy().flatten().astype(int))
            print('flattening done')

            # Calculate losses for da-net
            bce_da = bce_loss(all_da, masks)
            print('bce_da:', bce_da)

            # Backward pass and optimization
            bce_da.backward()
            optimizer_danet.step()

            # Calculate losses for lf-net
            l1 = l1_loss(all_lf, masks)
            print('l1:', l1)

            epe = epe_loss_fn(all_lf, masks)
            print('epe:', epe)

            loss_lf = bce_da + l1 + epe
            print('loss_lf:', loss_lf)

            # Backward pass and optimization
            loss_lf.backward()
            optimizer_lfnet.step()

            # Update weights
            optimizer_lfnet.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_lf.item()}")

        # save logging to file
        with open(logger_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_lf.item()}\n")

    # Save the trained model_lfnet and metrics
    metrics = {
        'loss_lf': loss_lf.item(),
        'loss_da': bce_da.item()
    }
    files = {}

    trained_model_path = make_file(result_id, 'model_lfnet.pkl')
    torch.save(model_lfnet.state_dict(), trained_model_path)
    with open(trained_model_path, 'rb') as f:
        files[trained_model_path] = f.read()
    with open(logger_path, 'rb') as f:
        files[logger_path] = f.read()

    print('DONE')

    return TrainResults(
        pretrained_model=trained_model_path.split('/')[-1],
        metrics=metrics,
        files=files
    )
