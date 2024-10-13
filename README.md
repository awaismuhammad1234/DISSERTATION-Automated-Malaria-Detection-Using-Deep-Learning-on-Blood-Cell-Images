# Automated Malaria Detection Using Deep Learning on Blood Cell Images

## Project Overview

This project aims to develop a Convolutional Neural Network (CNN) model for detecting malaria from blood cell images. The goal is to classify blood cell images into parasitized (infected) or uninfected (non-infected) categories to provide an automated, efficient, and accurate solution for malaria detection. This can help healthcare professionals, especially in malaria-endemic regions, by speeding up diagnosis and reducing the workload on laboratory technicians.

### Research Question
How effectively can deep learning detect malaria in blood cell images, and how does it compare to traditional methods in terms of accuracy?

### Project Objectives
1. **Develop a CNN Model**: Build and deploy a CNN deep learning model to classify parasitized and uninfected blood cells.
2. **Model Training and Evaluation**: Train and test the model to achieve high accuracy.
3. **User Interface Development**: Develop a simple web app using Streamlit, where users can upload blood cell images and receive predictions.

### Dataset

The dataset used for this project is publicly available on Kaggle and contains **27,558 blood cell images** categorized into two classes:
- **Parasitized**: Blood cells infected with malaria parasites.
- **Uninfected**: Blood cells not infected with malaria.

**Dataset link**: [Cell Images for Malaria Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

### Key Features
- **Model Architecture**: A CNN-based architecture, fine-tuned and trained to differentiate parasitized blood cells from uninfected ones.
- **Accuracy**: The model will be evaluated based on accuracy, precision, recall, and other metrics to ensure reliable performance.
- **Web Application**: A user-friendly web app using **Streamlit**, allowing users to upload their own blood cell images and get predictions.
