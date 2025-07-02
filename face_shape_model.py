import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torchvision.transforms as transforms

class FaceShapeModel(nn.Module):
    def __init__(self, num_shape_classes=5):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.gender_head = nn.Linear(feat_dim, 2)  # Gender classification
        self.shape_head = nn.Linear(feat_dim, num_shape_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.gender_head(x), self.shape_head(x)

class FaceShapeDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = FaceShapeModel()
        model_state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(model_state, strict=False)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.shape_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.gender_classes = ['Male', 'Female']

    def detect_face_shape(self, image_path):
        """
        Detect face shape and gender from an image file
        Args:
            image_path: Path to the image file
        Returns:
            tuple: (face_shape, face_confidence, gender, gender_confidence)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            gender_out, shape_out = self.model(image_tensor)
            
            # Process gender prediction
            gender_probabilities = torch.softmax(gender_out, dim=1)
            gender_pred_idx = gender_probabilities.argmax(1).item()
            gender_confidence = gender_probabilities[0][gender_pred_idx].item()
            gender_pred = self.gender_classes[gender_pred_idx]
            
            # Process shape prediction
            shape_probabilities = torch.softmax(shape_out, dim=1)
            shape_pred_idx = shape_probabilities.argmax(1).item()
            shape_confidence = shape_probabilities[0][shape_pred_idx].item()
            shape_pred = self.shape_classes[shape_pred_idx]

        return shape_pred, shape_confidence, gender_pred, gender_confidence
