import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import shutil

MODEL_PATH = "./waste_classifier_finetuned.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "./waste_classifier.pth"

HARD_EXAMPLES_DIR = "./hard_examples/"
IMAGE_SIZE = 260
CLASS_NAMES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

def create_model(num_classes):
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

import argparse

def parse_args():
    """Parses command-line arguments for the image path."""
    parser = argparse.ArgumentParser(description="Predict the waste category of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file for prediction.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    IMAGE_TO_PREDICT = args.image_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        exit()
        
    model.to(device)
    model.eval()

    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"File not found at '{IMAGE_TO_PREDICT}'. Please check the path.")
    else:
        try:
            image = Image.open(IMAGE_TO_PREDICT).convert('RGB') 
            image_tensor = transform(image).unsqueeze(0).to(device) 

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
            predicted_class = CLASS_NAMES[pred_idx.item()]
            confidence_percent = confidence.item() * 100 

            print(f"\nImage: '{os.path.basename(IMAGE_TO_PREDICT)}'")
            print(f"Prediction: '{predicted_class}'")
            print(f"Confidence: {confidence_percent:.2f}%\n")

            while True:
                feedback = input("Was this prediction correct? (y/n): ").lower() 
                if feedback in ['y', 'n']:
                    break
                print("Invalid input. Please enter 'y' or 'n'.")

            if feedback == 'n': 
                print("\nWhich was the correct class?")
                for i, name in enumerate(CLASS_NAMES): 
                    print(f"  {i+1}: {name}")
                
                while True: 
                    try:
                        correct_idx_input = int(input(f"Enter the number (1-{len(CLASS_NAMES)}): "))
                        if 1 <= correct_idx_input <= len(CLASS_NAMES):
                            correct_class_name = CLASS_NAMES[correct_idx_input - 1]
                            break
                        else:
                            print("Number out of range. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                dest_dir = os.path.join(HARD_EXAMPLES_DIR, correct_class_name)
                os.makedirs(dest_dir, exist_ok=True) 
                
                file_name = os.path.basename(IMAGE_TO_PREDICT) 
                dest_path = os.path.join(dest_dir, file_name) 
                
                shutil.copy(IMAGE_TO_PREDICT, dest_path) 
                print(f"Image saved to '{dest_path}' for future training.")
            else: 
                print("Prediction logged as correct.")
                
        except Exception as e: 
            print(f"An error occurred during prediction: {e}")