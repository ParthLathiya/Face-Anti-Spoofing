# Face Anti-Spoofing Mini Project (HONORS OEP)

This project is designed to detect and differentiate between real and fake faces using a face anti-spoofing model trained with the YOLO library. It includes modules for data collection, data splitting, model training, and real-time inference.

## Features
- **Data Collection**: Captures labeled face images from a webcam with a blur detection threshold.
- **Data Splitting**: Organizes collected data into training, validation, and testing sets.
- **Model Training**: Trains a YOLO model on the prepared dataset.
- **Real-Time Inference**: Detects and classifies faces as "real" or "fake" in real-time using a webcam.

## Directory Structure
The following directory structure is used in this project:

```
Face Anti-Spoofing(mini project)
|-- HONORS OEP
    |-- Data Collect
    |   |-- [collected_images].jpg
    |   |-- [collected_labels].txt
    |
    |-- Data Split
    |   |-- train
    |   |   |-- images
    |   |   |   |-- [training_images].jpg
    |   |   |-- labels
    |   |       |-- [training_labels].txt
    |   |
    |   |-- val
    |   |   |-- images
    |   |   |   |-- [validation_images].jpg
    |   |   |-- labels
    |   |       |-- [validation_labels].txt
    |   |
    |   |-- test
    |   |   |-- images
    |   |   |   |-- [test_images].jpg
    |   |   |-- labels
    |   |       |-- [test_labels].txt
    |
    |-- best.pt  # Trained YOLO model file
    |
    |-- data.yaml  # Configuration file for training and validation
```

## Getting Started

### Prerequisites
- Python 3.7+
- Required libraries:
  - `opencv-python`
  - `cvzone`
  - `ultralytics`
  - `numpy`
  - `shutil`

Install dependencies using:
```bash
pip install opencv-python cvzone ultralytics numpy
```

### Steps

1. **Data Collection**
   Run the following script to collect face images:
   ```bash
   python data_collection.py
   ```
   - Press `q` to stop the data collection process.
   - Images and labels will be saved in `Data Collect`.

2. **Data Splitting**
   Split the collected data into training, validation, and test sets:
   ```bash
   python data_splitting.py
   ```
   - The split data will be saved in `Data Split`.
   - A `data.yaml` file will also be generated for training configuration.

3. **Model Training**
   Train the YOLO model using the prepared dataset:
   ```bash
   python model_training.py
   ```
   - The trained model (`best.pt`) will be saved in the project directory.

4. **Real-Time Inference**
   Perform real-time face anti-spoofing detection:
   ```bash
   python main.py
   ```
   - Press `q` to exit the inference loop.

## Notes
- Ensure the webcam is properly connected for data collection and inference.
- Modify the confidence threshold or blur detection settings in the scripts as needed.
- Use `best.pt` as the default YOLO model for inference.

## Future Enhancements
- Integrate additional anti-spoofing techniques (e.g., texture analysis, depth estimation).
- Support for multiple cameras.
- Deployment as a standalone application.

## Author
Parth Lathiya

~This is a part of one of mini projects.
