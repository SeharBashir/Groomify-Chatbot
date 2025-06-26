import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class SkinToneDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.classes = ['dark', 'light', 'medium', 'tan']  # Based on the notebook
        
        # Data transforms matching the training setup
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Load the pre-trained MobileNetV2 model"""
        try:
            # Create the same model architecture as in training
            model = models.mobilenet_v2(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.7),
                nn.Linear(num_ftrs, 50),
                nn.Dropout(0.5),
                nn.Linear(50, 4)  # 4 classes: dark, light, medium, tan
            )
            
            # Load the trained weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading skin tone model: {e}")
            return None

    def detect_skin_tone(self, image_path):
        """
        Detect skin tone from an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_skin_tone, confidence_score)
        """
        if self.model is None:
            return "unknown", 0.0
            
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
                
                predicted_skin_tone = self.classes[predicted_class_idx]
                
            return predicted_skin_tone, confidence
            
        except Exception as e:
            print(f"Error detecting skin tone: {e}")
            return "unknown", 0.0

    def get_skin_tone_advice(self, skin_tone):
        """Get skin tone specific advice"""
        advice = {
            'light': {
                'description': 'Fair skin with cool undertones',
                'tips': [
                    'Use broad-spectrum SPF 30+ daily',
                    'Choose cool-toned makeup shades',
                    'Avoid harsh exfoliation',
                    'Use gentle, hydrating products'
                ]
            },
            'medium': {
                'description': 'Medium skin with warm or neutral undertones',
                'tips': [
                    'Use SPF 25+ for daily protection',
                    'Warm or neutral makeup tones work well',
                    'Regular gentle exfoliation is beneficial',
                    'Vitamin C serums help maintain glow'
                ]
            },
            'tan': {
                'description': 'Tan skin with warm undertones',
                'tips': [
                    'Use SPF 20+ to prevent further darkening',
                    'Golden and bronze makeup tones are flattering',
                    'Hydrating masks help maintain elasticity',
                    'Antioxidant-rich products are beneficial'
                ]
            },
            'dark': {
                'description': 'Deep skin with rich undertones',
                'tips': [
                    'SPF is still important for protection',
                    'Rich, warm makeup tones complement well',
                    'Focus on hyperpigmentation prevention',
                    'Gentle brightening products help even tone'
                ]
            }
        }
        
        return advice.get(skin_tone, {
            'description': 'Beautiful unique skin tone',
            'tips': ['Embrace your natural beauty', 'Use products suitable for your skin type']
        })
