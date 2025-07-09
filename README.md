<<<<<<< HEAD
# Groomify AI Chatbot

Groomify AI Chatbot is an intelligent virtual assistant that provides personalized beauty and grooming recommendations based on facial analysis, hair type detection, gender detection, and skin type analysis. The system uses advanced deep learning models to provide tailored advice for hairstyles and skincare products.

## Features

- **Face Shape Analysis**: Detects face shape with high accuracy
- **Hair Style Detection**: Classifies hair styles into 5 categories (Straight, Wavy, Curly, Dreadlocks, Kinky)
- **Gender Detection**: Provides gender classification
- **Skin Type Analysis**: Determines skin type (Dry, Normal, Oily)
- **Personalized Recommendations**: 
  - Hairstyle recommendations based on face shape, gender, and current hair type
  - Skincare product recommendations based on skin type
- **Interactive Chat Interface**: Natural language interaction for grooming advice

## Model Performance

- **Hair Style Detection**: 93% accuracy with balanced performance across all hair types
- **Skin Type Detection**: 87% accuracy across different skin types
- **Face Shape Detection**: 72.85% combined validation accuracy
- **Gender Detection**: ~95% validation accuracy

## Technical Stack

- **Backend**: Flask
- **AI/ML**: PyTorch, OpenCV, scikit-learn
- **NLP**: spaCy
- **Data Processing**: Pandas, NumPy
- **Image Processing**: Pillow, Albumentations

## Directory Structure

```
├── app.py                      # Main Flask application
├── chatbot.py                  # Chatbot logic
├── models/                     # Pre-trained model weights
├── datasets/                   # Training and recommendation data
├── templates/                  # HTML templates
└── requirements.txt           # Python dependencies
```

## Models Architecture

The system uses state-of-the-art deep learning models:
- Face Shape & Gender: Multi-task learning with ResNet
- Hair Style: Fine-tuned ResNet50
- Skin Type: Modified ResNet architecture

## Dataset Sources

- Hair Type Dataset: [Kaggle Hair Type Dataset](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset)
- Face Shape Dataset: Custom dataset with multiple face shapes
- Skin Type Dataset: Combination of public and custom collected data

## Installation & Setup

See `how_to_run.txt` for detailed installation and running instructions.
=======
# Groomify-Chatbot
>>>>>>> 4338e50c657f30766938af138460f5ff24ef58dd
