# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import random
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Create a directory to save the images
os.makedirs('synthetic_documents', exist_ok=True)

# List of sample texts in different languages
sample_texts = [
    (u"Driver's License", "English"),           # English
    (u"许可证", "Chinese"),                      # Chinese
    (u"رخصة قيادة", "Arabic"),                  # Arabic
    (u"Permis de conducere", "Romanian"),       # Romanian
    (u"ڈرائیور کا لائسنس", "Urdu"),             # Urdu
    (u"Πιστοποιητικό", "Greek"),                # Greek
    (u"Licencia de Conducir", "Spanish"),       # Spanish
    (u"Permesso di guida", "Italian"),          # Italian
    (u"Carteira de Motorista", "Portuguese"),   # Portuguese
    (u"Permis de conduire", "French"),          # French
    (u"ใบขับขี่", "Thai"),                      # Thai
    (u"운전 면허증", "Korean"),                   # Korean
    (u"Водительское удостоверение", "Russian"), # Russian
]

# Random data to add to the documents
random_data = [
    u"123456789",                  # Random number
    u"01/01/2022",                 # Random date
    u"John Doe",                   # Random name
    u"AB123456",                   # Random ID
    u"123 Main St",                # Random address
    u"987654321",                  # Another random number
    u"02/02/2023",                 # Another random date
    u"Jane Smith",                 # Another random name
    u"CD789012",                   # Another random ID
    u"456 Elm St",                 # Another random address
]

# Image sizes and font sizes
IMAGE_SIZES = {
    'license': (850, 540),
    'passport': (1250, 875),
    'a4': (2480, 3508),
}

FONT_SIZES = {
    'license': (32, 20),
    'passport': (48, 28),
    'a4': (72, 36),
}

# Function to load font
def load_font(font_size):
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",  # Noto Sans
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # DejaVu Sans
        "/Library/Fonts/Arial Unicode.ttf",                      # Arial Unicode (macOS)
        "C:\\Windows\\Fonts\\arialuni.ttf",                      # Arial Unicode (Windows)
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    # Fallback to default font
    return ImageFont.load_default()

# Function to create a synthetic document
def create_synthetic_document(text, language, doc_type='license'):
    image_size = IMAGE_SIZES.get(doc_type, (850, 540))
    font_size, smaller_size = FONT_SIZES.get(doc_type, (32, 20))
    
    # Create a white background image using PIL
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Load font
    font = load_font(font_size)
    smaller_font = load_font(smaller_size)
    
    # Add main text
    x_margin = 20
    y_margin = 20
    draw.text((x_margin, y_margin), text, fill='black', font=font)
    
    # Add all sample texts in different languages
    y_offset = y_margin + font_size + 10
    for sample_text, lang in sample_texts:
        if lang != language:
            draw.text((x_margin, y_offset), sample_text, fill='black', font=smaller_font)
            y_offset += smaller_size + 5
    
    # Add random data
    x_offset = image_size[0] // 2
    y_offset = y_margin + font_size + 10
    for data in random_data:
        draw.text((x_offset, y_offset), data, fill='black', font=smaller_font)
        y_offset += smaller_size + 5
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv

# Function to apply perspective transformation
def apply_perspective_transform(img):
    rows, cols, ch = img.shape
    
    # Original points
    pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
    
    # Perturbed points
    margin = int(min(rows, cols) * 0.1)  # 10% margin
    pts2 = pts1 + np.random.uniform(-margin, margin, pts1.shape).astype(np.float32)
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply the warp perspective
    dst = cv2.warpPerspective(img, M, (cols, rows), borderValue=(255,255,255))
    
    return dst

# Function to rotate, apply perspective transform, and save images
def transform_and_save_image(img_array, angle, save_path):
    # Rotate image
    rotated_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    rotated_img = rotated_img.rotate(angle, expand=True, fillcolor='white')
    rotated_img = np.array(rotated_img)
    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
    
    # Apply perspective transform
    transformed_img = apply_perspective_transform(rotated_img)
    
    # Save the transformed image
    cv2.imwrite(save_path, transformed_img)

# Generate synthetic documents and augment with transformations
doc_types = ['license', 'passport', 'a4']
for doc_type in tqdm(doc_types, desc='Processing document types'):
    for i in tqdm(range(100), desc=f'Generating {doc_type} documents', leave=False):
        text, language = random.choice(sample_texts)
        img = create_synthetic_document(text, language, doc_type)
        base_filename = f'{language}_{doc_type}_{i}'
        # Save original image
        cv2.imwrite(f'synthetic_documents/{base_filename}_rot0.png', img)
        
        # Generate transformations directly from the original image
        for j in range(10):
            angle = random.uniform(-180, 180)
            angle_str = f"{angle:.2f}"
            save_path = f'synthetic_documents/{base_filename}_rot{angle_str}.png'
            transform_and_save_image(img, angle, save_path)

# Custom Dataset
class DocumentDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform

        for filename in os.listdir(img_dir):
            if filename.endswith('.png'):
                angle = float(filename.split('_rot')[-1].replace('.png', ''))
                img_path = os.path.join(img_dir, filename)
                self.img_labels.append((img_path, angle))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, angle = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        # Compute sine and cosine
        sin_angle = math.sin(angle_rad)
        cos_angle = math.cos(angle_rad)
        
        # Return image and sine/cosine labels
        return image.float(), torch.tensor([sin_angle, cos_angle], dtype=torch.float32)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = DocumentDataset('synthetic_documents', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model with Spatial Transformer Network
class STNOrientationModel(nn.Module):
    def __init__(self):
        super(STNOrientationModel, self).__init__()
        # Load pre-trained ResNet18 model
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Flatten()  # Add flatten layer to handle dimensions automatically
        )

        # Calculate the output size of localization network
        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_output = self.localization(dummy_input)
        loc_output_size = dummy_output.shape[1]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Modify the final layer to output sine and cosine
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output sine and cosine
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x = nn.functional.grid_sample(x, grid, align_corners=False)
        
        return x

    def forward(self, x):
        # Transform the input
        x = self.stn(x)
        # Forward pass through the base model
        x = self.base_model(x)
        return x

model = STNOrientationModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in tqdm(range(num_epochs), desc='Training epochs'):
    model.train()
    running_loss = 0.0
    total = 0
    total_angle_error = 0.0

    for images, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        images = images.float()
        labels = labels.float()
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        
        # Compute predicted angles
        pred_sin = outputs[:, 0].detach().numpy()
        pred_cos = outputs[:, 1].detach().numpy()
        pred_angles = np.arctan2(pred_sin, pred_cos) * (180 / np.pi)
        
        # Compute true angles
        true_sin = labels[:, 0].numpy()
        true_cos = labels[:, 1].numpy()
        true_angles = np.arctan2(true_sin, true_cos) * (180 / np.pi)
        
        # Compute angular error
        angle_errors = np.abs(pred_angles - true_angles)
        angle_errors = np.minimum(angle_errors, 360 - angle_errors)  # Account for periodicity
        total_angle_error += np.sum(angle_errors)
    
    avg_loss = running_loss / total
    avg_angle_error = total_angle_error / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Avg Angle Error: {avg_angle_error:.2f} degrees')
