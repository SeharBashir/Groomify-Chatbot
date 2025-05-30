import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np

class SkinTypeDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize with ImageNet weights first
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: dry, normal, oily
        
        # Load the trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Remove fc layer and problematic keys from state dict
        keys_to_remove = ['fc.weight', 'fc.bias', 'fc.in_features.weight', 'fc.in_features.bias']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
            
        # Load the state dict without fc layer
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations - exactly as used in training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Define label mappings
        self.index_label = {0: "dry", 1: "normal", 2: "oily"}
        self.label_index = {"dry": 0, "normal": 1, "oily": 2}

    def detect_skin_type(self, image_path):
        """
        Detect skin type from an image
        Args:
            image_path: Path to the image file
        Returns:
            tuple: (predicted_skin_type, confidence_score)
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img = self.transform(np.array(img))
            img = img.unsqueeze(0)  # Add batch dimension
            img = img.to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_label = self.index_label[predicted.item()]
                confidence_score = confidence.item()

            return predicted_label, confidence_score

        except Exception as e:
            print(f"Error in skin type detection: {str(e)}")
            return "unknown", 0.0