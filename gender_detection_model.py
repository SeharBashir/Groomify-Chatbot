from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

class GenderDetector:
    def __init__(self, model_name="rizvandwiki/gender-classification-2"):
        """
        Initialize the gender detector
        Args:
            model_name: Name of the model
        """
        try:
            # Load the processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading gender detection model: {str(e)}")
            raise e

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
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process the image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get predicted class and confidence
            predicted_label = logits.argmax(-1).item()
            predicted_gender = self.model.config.id2label[predicted_label]
            
            # Calculate confidence (softmax probability)
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities[0][predicted_label].item()

            return predicted_gender, confidence

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None
