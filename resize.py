import os
from PIL import Image
import shutil

INPUT_DIR = 'dataset-original' 

OUTPUT_DIR = 'data_sorted/'

IMAGE_SIZE = (260, 260) 

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)
print(f"Output directory created at: {OUTPUT_DIR}")

try:
    categories = os.listdir(INPUT_DIR)
except FileNotFoundError:
    print(f"ERROR: The input directory '{INPUT_DIR}' was not found.")
    
    categories = []

total_images_processed = 0

for category in categories:
    input_category_path = os.path.join(INPUT_DIR, category)
    output_category_path = os.path.join(OUTPUT_DIR, category)
    
    if not os.path.isdir(input_category_path):
        continue
        
    print(f"\nProcessing category: '{category}'")
    
    os.makedirs(output_category_path, exist_ok=True)
    
    image_counter = 1
    
    for filename in sorted(os.listdir(input_category_path)):
        input_image_path = os.path.join(input_category_path, filename)
        
        try:
            with Image.open(input_image_path) as img:
                img_rgb = img.convert('RGB')
                
                img_resized = img_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                new_filename = f"{category.lower()}_{image_counter}.jpg"
                output_image_path = os.path.join(output_category_path, new_filename)
                
                img_resized.save(output_image_path, 'JPEG')
                
                image_counter += 1
                
        except Exception as e:
            print(f"Could not process file '{filename}': {e}")
            
    processed_count = image_counter - 1
    total_images_processed += processed_count
    print(f"Processed and saved {processed_count} images.")

print(f"Total images processed: {total_images_processed}")
print(f"Data is now ready for training in '{OUTPUT_DIR}'")