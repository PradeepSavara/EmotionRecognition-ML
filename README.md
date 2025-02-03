📂 Emotion-Recognition
│── 📂 data
│   └── fer2013.csv               # FER-2013 dataset
│── 📂 model
│   └── emotion_model.h5           # Trained emotion recognition model
│── 📂 src
│   ├── train_model.py             # Model training script
│   ├── emotion_recognition.py     # Real-time emotion detection script
│── README.md                      # Project documentation

Emotion Recognition using FER-2013:
This project uses Convolutional Neural Networks (CNNs) to classify facial emotions from images. It is trained on the FER-2013 dataset and uses OpenCV for real-time emotion detection via webcam.
Dataset
The FER-2013 dataset contains 35,887 grayscale images (48x48 pixels) labeled into 7 emotion categories:
Angry,
Disgust,
Fear,
Happy,
Sad,
Surprise,
Neutral,
