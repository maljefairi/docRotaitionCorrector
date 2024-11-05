# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Create a directory to save the images
os.makedirs('synthetic_documents', exist_ok=True)

# List of sample texts in different languages
sample_texts = [
    (u"Driver's License", "English"),           # English
    (u"许可证", "Chinese"),                      # Chinese
    (u"رخصة قيادة", "Arabic"),                  # Arabic
    (u"Permis de conducere", "Romanian"),        # Romanian
    (u"ڈرائیور کا لائسنس", "Urdu"),          # Urdu
    (u"Πιστοποιητικό", "Greek"),              # Greek
    (u"Licencia de Conducir", "Spanish"),       # Spanish
    (u"Permesso di guida", "Italian"),          # Italian
    (u"Carteira de Motorista", "Portuguese"),      # Portuguese
    (u"Permis de conduire", "French"),         # French
    (u"ใบขับขี่", "Thai"),                   # Thai
    (u"운전 면허증", "Korean"),                 # Korean
    (u"Водительское удостоверение", "Russian"),  # Russian
]

# List of random data to add to the documents
random_data = [
    u"123456789",                  # Random number
    u"01/01/2022",                 # Random date
    u"John Doe",                   # Random name
    u"AB123456",                   # Random ID
    u"123 Main St",                # Random address
    u"987654321",                  # Another random number
    u"02/02/2023",                # Another random date
    u"Jane Smith",                 # Another random name
    u"CD789012",                   # Another random ID
    u"456 Elm St",                 # Another random address
]

# Function to create a synthetic document
def create_synthetic_document(text, language, doc_type='license'):
    # Set image size based on document type
    if doc_type == 'license':
        image_size = (850, 540)  # Standard US driver's license size (3.375" x 2.125" at 250 DPI)
    elif doc_type == 'passport':
        image_size = (1250, 875)  # Standard passport size (4.92" x 3.46" at 250 DPI)
    else:  # A4 document
        image_size = (2480, 3508)  # A4 size at 300 DPI

    # Create a white background image using PIL
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # List of font paths to try for different operating systems and languages
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",  # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux CJK
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",  # Linux Arabic
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",  # Linux Thai
        "/Library/Fonts/Arial Unicode.ttf",  # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        "C:\\Windows\\Fonts\\arialuni.ttf",  # Windows Unicode
    ]
    
    # Adjust font size based on document type
    if doc_type == 'license':
        font_size = 32
        smaller_size = 20
    elif doc_type == 'passport':
        font_size = 48
        smaller_size = 28
    else:  # A4 document
        font_size = 72
        smaller_size = 36
    
    # Try to load fonts until one works
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    # If no font was loaded, try to download and use Noto Sans
    if font is None:
        try:
            from urllib.request import urlretrieve
            font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
            font_path = "NotoSans-Regular.ttf"
            if not os.path.exists(font_path):
                urlretrieve(font_url, font_path)
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text size for main text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate center position for main text
    x = (image_size[0] - text_width) // 2
    y = 20  # Place main text at top
    
    # Add main text
    draw.text((x, y), text, fill='black', font=font)
    
    # Use smaller font for additional text
    smaller_font = ImageFont.truetype(font_path, smaller_size)
    
    # Add all sample texts in different languages
    y_offset = y + text_height + 20
    x_margin = 20
    for sample_text, lang in sample_texts:
        if lang != language:  # Skip the main language
            draw.text((x_margin, y_offset), sample_text, fill='black', font=smaller_font)
            y_offset += smaller_size + 10
    
    # Add all random data
    x_margin = image_size[0] // 2 + 20
    y_offset = y + text_height + 20
    for data in random_data:
        draw.text((x_margin, y_offset), data, fill='black', font=smaller_font)
        y_offset += smaller_size + 10
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv

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
        
        # Normalize angle to be between 0 and 1
        normalized_angle = (angle + 180) / 360

        if self.transform:
            image = self.transform(image)

        return image, normalized_angle

# Generate synthetic documents
doc_types = ['license', 'passport', 'a4']
for doc_type in doc_types:
    for i in range(10):
        text, language = random.choice(sample_texts)
        img = create_synthetic_document(text, language, doc_type)
        # Save original image with its true rotation (0 degrees)
        cv2.imwrite(f'synthetic_documents/{language}_{doc_type}_{i}_rot0.png', img)
        # Save with compression
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        cv2.imwrite(f'synthetic_documents/{language}_{doc_type}_{i}_rot0.png', img, encode_param)

# Augment the dataset with random rotations
for filename in tqdm(os.listdir('synthetic_documents')):
    if filename.endswith('rot0.png'):
        img_path = os.path.join('synthetic_documents', filename)
        img = Image.open(img_path)
        base_name = filename[:-8]  # Remove '_rot0.png'

        # Generate 10 random rotations between -180 and 180 degrees
        for i in range(10):
            # Generate random rotation angle
            angle = random.uniform(-180, 180)
            # Round to 2 decimal places for filename
            angle_str = f"{angle:.2f}"
            rotated = img.rotate(angle, expand=True)
            rotated.save(f'synthetic_documents/{base_name}_rot{angle_str}.png')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = DocumentDataset('synthetic_documents', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the CNN model
class OrientationCNN(nn.Module):
    def __init__(self):
        super(OrientationCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output layer for regression
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = OrientationCNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0

    for images, angles in dataloader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        total += angles.size(0)

    avg_loss = running_loss/total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
