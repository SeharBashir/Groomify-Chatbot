import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms

class HairStyleModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.softmax(x, dim=1)

class HairStyleDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = HairStyleModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.hair_classes = ['Straight', 'Wavy', 'Curly', 'Dreadlocks', 'Kinky']

    def detect_hair_style(self, image_path):
        """
        Detect hair style from an image file
        Args:
            image_path: Path to the image file
        Returns:
            str: Predicted hair style
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            hair_out = self.model(image_tensor)
            pred_idx = hair_out.argmax(1).item()
            confidence = hair_out[0][pred_idx].item()
            hair_pred = self.hair_classes[pred_idx]

        return hair_pred, confidence
