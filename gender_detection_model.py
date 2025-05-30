import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomClassifier_Custom(nn.Module):
    def __init__(self, in_features, weights_arr, num_classes):
        super(CustomClassifier_Custom, self).__init__()
        self.weights_arr = weights_arr
        #-----------------------------------------------------
        self.fc1 = nn.Linear(in_features, self.weights_arr[0])  
        self.bn1 = nn.BatchNorm1d(self.weights_arr[0])
        #-----------------------------------------------------
        if (len(self.weights_arr) >= 2):
            self.layer_custom = nn.Sequential()
            for i in range(len(self.weights_arr)-1):
                self.layer_custom.add_module(f"conv_{i}", nn.Linear(self.weights_arr[i],self.weights_arr[i+1]))
                self.layer_custom.add_module(f"bn_{i}", nn.BatchNorm1d(self.weights_arr[i + 1])) 
                self.layer_custom.add_module(f"relu_{i}",nn.ReLU())
                self.layer_custom.add_module(f"drop_{i}",nn.Dropout(p=0.5))
        #-----------------------------------------------------        
        self.Dropuot = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(self.weights_arr[-1], num_classes)  
                    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.Dropuot(x)
        if (len(self.weights_arr) >= 2):
            x = self.layer_custom(x)
        x = self.fc2(x)
        return x

class GenderDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize model architecture exactly as in training from s3.py
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet18.parameters():
            param.requires_grad = False
        resnet18.fc = CustomClassifier_Custom(resnet18.fc.in_features, [1000], num_classes=5)  # Match the exact training setup
        
        # Load the model weights
        self.model = resnet18
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Define image preprocessing matching the training
        self.transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

        # Class labels exactly matching training data from s3.py
        self.class_labels = {0: 'Men', 1: 'Women'}

    def detect_gender(self, image_path):
        """
        Detect gender from an image file
        Args:
            image_path: Path to the image file
        Returns:
            str: Predicted gender
            float: Confidence score
        """
        try:
            # Load and preprocess image exactly as in training
            image = np.array(Image.open(image_path).convert('RGB'))
            augmented = self.transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                # Apply sigmoid as used in training
                probabilities = torch.sigmoid(outputs)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            return self.class_labels[pred_class], confidence

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None
