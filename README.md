# AI in Medicine with MedMNIST

Welcome to my repository dedicated to exploring the application of Artificial Intelligence in the field of medicine. This repository showcases a series of projects I developed as part of a course, focusing on leveraging the MedMNIST dataset to address various medical imaging challenges.

## About MedMNIST

MedMNIST is a large-scale dataset of biomedical images, designed to provide a standardized benchmark for automated machine learning models in medical image analysis. It offers a wide variety of datasets that span different modalities and scales, including but not limited to, pathology, dermatology, and radiology images. For more information, visit [MedMNIST Homepage](https://medmnist.com/).

## Projects Overview

This repository contains several projects that demonstrate the application of AI techniques in medical imaging analysis. The projects are categorized into three main areas:

### 1. Classification Models

- **Objective**: Develop models to accurately classify medical images into various categories, such as disease identification or anomaly detection.
- **Techniques Used**: Deep learning models including CNNs (Convolutional Neural Networks) applied on various subsets of the MedMNIST dataset.

### 2. Segmentation

- **Objective**: Implement segmentation models to identify and delineate specific structures or regions within medical images.
- **Techniques Used**: U-Net architecture and other segmentation algorithms to perform pixel-wise classification, enhancing the accuracy of medical diagnosis.

### 3. Differential Privacy in AI

- **Objective**: Explore the implementation of differential privacy techniques to protect patient data while training AI models.
- **Achievements**: Successfully tested models with differential privacy to ensure data confidentiality without compromising the model's performance.

### 4. Explainability
- **Objective**: Learn about techniques for making the predictions of convolutional neural networks interpretable, with a focus on Gradient-weighted Class Activation Mapping (Grad-CAM).
- **Achievements**: Successfully visual demonstration on which input features are responsible for the model's prediction

## Technologies Used

- Python
- [PyTorch](https://pytorch.org/)
- [Opacus](https://opacus.ai)
- [PythonGradCam](https://github.com/jacobgil/pytorch-grad-cam)

## Flashcards for AI in Medicine

In addition to the Jupyter notebooks, this repository includes a set of flashcards optimized for use with Obsidian.
These flashcards cover key concepts, terminologies, and practices in AI in medicine, making them a perfect tool for quick revision and reinforcing your learning. Whether you're a student, a healthcare professional, or an AI enthusiast, these flashcards are designed to help you grasp and retain the complex intricacies of AI in medicine effectively.

Also I recommend downloading the [`Automatic Table of Contents` plugin](https://github.com/johansatge/obsidian-automatic-table-of-contents) for a better structure.

## Getting Started

To get started with the notebooks and flashcards in this repository:

1. **Clone the Repository**: Clone this repository to your local machine to access the Jupyter notebooks and flashcards.

   ```
   git clone https://github.com/TimMachTUM/AI-in-medicine.git
   ```

2. **Install Requirements**: Make sure to install the required libraries by running:

   ```
   pip install -r requirements.txt
   ```

3. **Explore the Notebooks**: Navigate to the notebook directory and launch Jupyter Notebook or Jupyter Lab to start exploring:

   ```
   jupyter notebook
   ```

   or

   ```
   jupyter lab
   ```

4. **Import Flashcards into Obsidian**: To use the flashcards, import them into your Obsidian vault and start your learning journey.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.