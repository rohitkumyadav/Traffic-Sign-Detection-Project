# Traffic Sign Recognition AI

## Project Overview
The **Traffic Sign Recognition AI** is a deep learning-based system designed to recognize traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The project employs **Convolutional Neural Networks (CNNs)** for classification and is deployed as a web application using **Streamlit**.

## Project Components
### 1. Model Training (`main.ipynb`)
- The **training script** is implemented in Jupyter Notebook.
- It loads the GTSRB dataset, preprocesses the images, and trains a CNN model for classification.
- The trained model is saved as `traffic_sign_cnn_model.h5`.

### 2. Model Deployment (`model_deploy.py`)
- The **Streamlit application** allows users to upload an image of a traffic sign.
- The uploaded image is preprocessed and passed to the trained CNN model.
- The model predicts the traffic sign label and displays the result along with a confidence score.
- The application includes a mapping of class labels to real-world traffic signs.

## Features
- **Deep Learning Model**: Uses CNN for accurate traffic sign recognition.
- **User-Friendly Interface**: Built with Streamlit for ease of use.
- **Real-Time Predictions**: Processes and classifies images instantly.
- **Confidence Score**: Displays the model's confidence in its prediction.

## Usage Instructions
1. **Train the Model**
   - Open `main.ipynb`.
   - Execute the notebook to train the CNN model.
   - The trained model is saved as `traffic_sign_cnn_model.h5`.

2. **Run the Streamlit App**
   - Ensure `traffic_sign_cnn_model.h5` is in the working directory.
   - Execute the following command in the terminal:
     ```bash
     streamlit run model_deploy.py
     ```
   - Upload a traffic sign image and get the classification result.

## Dependencies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- PIL (Pillow)
- Streamlit

## Future Enhancements
- Improve model accuracy with data augmentation.
- Support additional traffic sign datasets.
- Deploy as a cloud-based API.
- Enhance the UI with interactive elements.

This project is a practical demonstration of AI in **real-world traffic systems** and can be further extended for **autonomous vehicle applications**.

