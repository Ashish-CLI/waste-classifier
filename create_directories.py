import os

def create_waste_classifier_directories():
    base_dir = "waste_classifer"
    
    data_dirs = ["hard_examples", "dataset-original"]
    
    categories = ["Hazardous", "Organic", "Recyclable", "Non-Recyclable"]
    
    for data_dir in data_dirs:
        for category in categories:
            path = os.path.join(base_dir, data_dir, category)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

if __name__ == "__main__":
    create_waste_classifier_directories()