# Traffic Light Detection

## Project Overview
The **Traffic Light Detection Project** is a machine learning-based system designed to accurately detect and classify traffic lights in real-time. This project employs **deep learning techniques** to enhance road safety, assist autonomous vehicles, and improve traffic management systems.

## Project Components
### 1. Model Training (`train_model.py`)
- The **training script** processes traffic light images and trains a deep learning model using a **Convolutional Neural Network (CNN)**.
- The dataset consists of labeled images of traffic lights in various conditions.
- The trained model is saved for real-time inference.

### 2. Model Deployment (`detect_traffic_lights.py`)
- The **detection module** processes live camera feeds or images.
- The model identifies traffic lights and classifies them as **red, yellow, or green**.
- Outputs include **bounding boxes** and real-time **traffic signal status**.

## Features
- **Real-Time Traffic Light Detection**: Classifies lights into red, yellow, or green.
- **Deep Learning Model**: Utilizes a pre-trained CNN for efficient recognition.
- **Robust Performance**: Works in various lighting and weather conditions.
- **Scalable**: Can be integrated into autonomous vehicle systems and smart traffic management applications.
- **Optimized Processing**: Ensures real-time inference with minimal latency.

## Installation
### Prerequisites
Ensure the following dependencies are installed:
- Python 3.x
- TensorFlow / PyTorch
- OpenCV
- NumPy
- Matplotlib (for visualization)

### Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rohitkumyadav/Traffic-Light-Detection-Project.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Traffic-Light-Detection-Project
   ```
3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the Detection Script:**
   ```bash
   python detect_traffic_lights.py
   ```
2. **Supported Input Sources:**
   - Live camera feed
   - Pre-recorded videos
   - Static images
3. **Output:**
   - Bounding boxes around detected traffic lights
   - Traffic signal classification

## Model Training
### Dataset
The model is trained on a dataset containing labeled images of traffic lights under various conditions (day, night, rain, fog, etc.).

### Training Process
To train the model or fine-tune an existing one:
```bash
python train_model.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

## Performance Evaluation
The model is evaluated based on:
- **Precision, Recall, and F1-score**
- **Confusion matrix analysis**
- **Testing on real-world datasets**

## Applications
- **Autonomous Vehicles**: Enhances navigation by recognizing traffic signals.
- **Smart Traffic Management**: Improves monitoring and regulation of traffic lights.
- **Surveillance Systems**: Assists in traffic law enforcement and road safety analysis.

## Future Enhancements
- Improve model accuracy with **data augmentation**.
- Expand dataset coverage to include more **real-world scenarios**.
- Deploy as a **cloud-based API** for integration into larger systems.
- Optimize for **edge computing devices** for low-latency processing.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new`).
3. Commit your changes.
4. Open a pull request for review.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support
If you find this project useful, consider **starring** the repository on GitHub.

---
Developed by [Rohit Kumar Yadav](https://github.com/rohitkumyadav)

