# Waste Classifier: Deep Learning for Sustainable Waste Management

This project provides a deep learning-based solution for classifying waste into four distinct categories: Hazardous, Organic, Recyclable, and Non-Recyclable. Leveraging the power of EfficientNet-B2, this system aims to contribute to more efficient waste sorting and sustainable environmental practices.

## Table of Contents
- [Setup](#setup)
- [Directory Structure](#directory-structure)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [How it Works](#how-it-works)
- [Model and Dataset](#model-and-dataset)

## Setup

### 1. Create Project Directories
To set up the necessary directory structure for your dataset and hard examples, run the `create_directories.py` script:

```bash
python create_directories.py
```

This script will set up the necessary folder structure:
- `waste_classifer/hard_examples/`: To store misclassified images for future analysis or retraining.
- `waste_classifer/dataset-original/`: Where you will place your raw waste images, organized by category.

Both `hard_examples` and `dataset-original` will contain subdirectories for 'Hazardous', 'Organic', 'Recyclable', and 'Non-Recyclable' waste categories.

### 2. Place Images in Categories
Organize your raw waste images into the respective category subdirectories within `waste_classifer/dataset-original/`. For example:
- `waste_classifer/dataset-original/Hazardous/image1.jpg`
- `waste_classifer/dataset-original/Organic/image2.png`
- `waste_classifer/dataset-original/Recyclable/image3.jpeg`
- `waste_classifer/dataset-original/Non-Recyclable/image4.jpg`

**Note:** The model trained for this project utilizes a combined dataset from various sources, including publicly available datasets on GitHub and Hugging Face, along with a small custom dataset. Details on these sources can be found in the [Model and Dataset](#model-and-dataset) section.

### 3. Install Dependencies
It is highly recommended to use a virtual environment to manage project dependencies.

First, create a virtual environment (if you don't have one):
```bash
python -m venv venv
```

Activate the virtual environment:
- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

Once the virtual environment is active, install the required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Resize Images
Before training, all images must be resized to a consistent dimension (260x260 pixels). The `resize.py` script handles this, taking images from `waste_classifer/dataset-original/` and saving the processed images to `waste_classifer/data_sorted/`.

Run the `resize.py` script:

```bash
python resize.py
```

## Training the Model

### 1. Train the Classifier
To train the waste classification model, execute the `train.py` script. This script will use the images from `waste_classifer/data_sorted/` and save the best performing model as `waste_classifier.pth` in the `waste_classifer/` directory.

```bash
python train.py
```

## Making Predictions

### 1. Predict on a New Image
To classify a new image, use the `prediction1.py` script and provide the path to the image as a command-line argument.

```bash
python prediction1.py path/to/your/image.jpg
```

The script will output the predicted waste category and confidence. It also includes an interactive prompt to gather feedback on the prediction's accuracy. If a prediction is incorrect, you can specify the correct class, and the image will be moved to the `hard_examples/` directory for that class, which can be used for future model improvements.

## How It Works: A Deep Dive into the Waste Classifier

This section provides a detailed explanation of the deep learning methodology and techniques employed in this waste classification project.

### The Brain: EfficientNet-B2
Fundamentally, our waste classifier employs what's known as **EfficientNet-B2**. It's like a super-efficient and precise convolutional neural network (CNN) – a form of AI well-suited to image tasks. EfficientNet models are smart because they increase their "brainpower" (depth, width, and image resolution) proportionally. We begin with an EfficientNet-B2 that has already learned a great deal from a massive image dataset (`IMAGENET1K_V1`), which provides us with an excellent head start. We fine-tune its very last layer to recognize exclusively our four types of waste: Hazardous, Organic, Recyclable, and Non-Recyclable.

### Getting the Data Ready
- **Coherent Images**: We utilize a specialized `StratifiedImageDataset` to ensure our images are loaded and labeled properly, keeping it all clean and organized for training.
- **Balanced Splits**: When we split our images into train and val sets, we utilize a method which ensures the ratio of each type of waste remains the same in both sets. It's really crucial so our model won't be biased towards classes with more images.
- **Image Makeovers (Augmentation)**:
    - **For Training**: To prepare our training images, we "makeup" them with `transforms.TrivialAugmentWide()`. That is, we rotate, shift, or shear them a little randomly. It's like exposing the model to the same thing from slightly different perspectives, so it can learn more effectively. All are resized to `260x260` pixels and normalized as well.
- **For Validation**: For validating how good the model is, we keep it straightforward. Images are resized and normalized, but no random makeovers, so we have a consistent and equitable testing.

### Handling Uneven Data: Smart Sampling
Occasionally, you may have tens of times as many "Organic" waste images as "Hazardous." This is class imbalance, and it will dupe the model into not noticing the less frequent ones. To correct this, we employ a `WeightedRandomSampler` during training. This nifty gadget ensures images from those less frequent categories are selected more frequently, compelling the model to notice all the waste types equally.

### The Training Process
- **The Optimizer (AdamW)**: This is similar to the coach of the model, teaching it how to learn from its error. We employ `AdamW` with certain configurations (`lr=1e-3`, `weight_decay=1e-2`) to assist it in discovering the optimal means of classifying images.
- **Dynamic Learning (OneCycleLR)**: Rather than a constant learning rate, we employ a `OneCycleLR` scheduler. It's similar to a clever accelerator that varies the learning rate during training, typically resulting in faster and improved outcomes.
- **Measurement Errors (CrossEntropyLoss with Label Smoothing)**: Our model trains by minimizing a "loss" value. We utilize `nn.CrossEntropyLoss` with `label_smoothing=0.1`. Label smoothing is a nice trick that keeps the model from being *too* sure, which actually makes it better at generalizing to new, unseen images.
- **Speed Boost (Mixed Precision Training)**: We utilize `torch.amp.autocast` and `torch.amp.GradScaler` to make training quicker and consume less memory. This is how the model achieves a combination of high and lower precision numbers, accomplishing the task faster without taking a hit on accuracy.
- **Training Rounds (Epochs)**: The model undergoes `15` complete rounds of training on the entire dataset.
- **Saving the Best**: We monitor how well the model performs on the validation set. Whenever it achieves a new personal best in terms of accuracy, that version of the model is saved. This ensures that the best-performing model is always available.

## Model

A pre-trained `waste_classifier.pth` model, trained on a combined dataset, will be uploaded on hugging face soon



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure the code adheres to the project's style guidelines.
4.  Write appropriate tests for your changes.
5.  Commit your changes (`git commit -m 'feat: Add new feature'`).
6.  Push to your branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request.
