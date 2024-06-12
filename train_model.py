import pickle
from typing import Any, Mapping
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import torch
import torch.nn as nn
from model1 import LFNet
from pymlab.train import TrainResults
from pymlab.utils import make_file

# EPE_LOSS class
class EPE_LOSS(nn.Module):
    def __init__(self):
        super(EPE_LOSS, self).__init__()

    def forward(self, output, mask):
        epe_loss = torch.mean(torch.sqrt(torch.sum((output - mask) ** 2, dim=1)))
        return epe_loss

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {
            'image': image,
            'mask': mask
        }
    

# Function to get all file paths from directory
def get_file_paths(directory, extensions):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
    return paths

def train_model(
    dataset_path: str,
    parameters: dict[str, Any],
    result_id: str,
    **kwargs
) -> TrainResults:
    
    image_size = parameters.get('image_size', 128)
    epochs = parameters.get('epochs', 10)
    learning_rate = parameters.get('learning_rate', 0.001)
    b1 = parameters.get('b1', 0.9)
    b2 = parameters.get('b2', 0.999)
    
    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load dataset
    image_dir = dataset_path + "/train"
    mask_dir = dataset_path + "/train_mask"

    image_paths = get_file_paths(image_dir, ['jpg'])
    mask_paths = get_file_paths(mask_dir, ['png'])
    print('image_paths:', image_paths)
    print('mask_paths:', mask_paths)


    dataset = CustomDataset(image_paths, mask_paths, transform)
    print('dataset:', dataset)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model, loss, and optimizer
    model = LFNet()
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    epe_loss_fn = EPE_LOSS()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2))

    logger_path = make_file(result_id, 'train.log')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            images = batch['image']
            masks = batch['mask']

            optimizer.zero_grad()
            print('before training img:', images.shape)
            outputs = model(images)
            print('outputs:', outputs.shape)
            outputsTransform = transforms.Compose([transforms.Resize((image_size, image_size))])
            outputs = outputsTransform(outputs)
            print('after training:', outputs.shape)

            # Calculate losses
            bce = bce_loss(outputs, masks)
            l1 = l1_loss(outputs, masks)
            epe = epe_loss_fn(outputs, masks)
            loss = bce + l1 + epe

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
        # save logging to file
        with open(logger_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}\n")        

    # Save the trained model
    metrics = {
        'loss': loss.item()
    }
    files = {}

    trained_model_path = make_file(result_id, 'model.pkl')
    # torch.save(model.state_dict(), "lfnet_model.pth")
    with open(trained_model_path, 'wb') as f:
        pickle.dump(model.state_dict, f)
    
    with open(trained_model_path, 'rb') as f:
        files[trained_model_path] = f.read()
    with open(logger_path, 'rb') as f:
        files[logger_path] = f.read()

    return TrainResults(
        pretrained_model=trained_model_path.split('/')[-1],
        metrics=metrics,
        files=files
    )
