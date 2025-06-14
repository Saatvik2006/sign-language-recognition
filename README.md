# 🤟 Real-Time Sign Language Recognition using Mediapipe, LSTM & OpenCV

This project uses **MediaPipe**, **OpenCV**, and a **deep learning model (LSTM)** to recognize American Sign Language gestures in real-time using a webcam. It predicts the following gestures:

- ✋ `hello`  
- 🙏 `thanks`  
- ❤️ `iloveyou`  
- ❓ `how are you`  
- 😔 `sorry`

---

## 🚀 Features
- Real-time video capture using OpenCV
- Hand, face, and pose landmark detection using MediaPipe Holistic
- LSTM-based deep learning model trained to classify gestures
- Smooth prediction logic with probability thresholds
- Custom probability bar UI for visual feedback

---

## 📁 Dataset
Collected using webcam:
- ~30 videos per action
- 30 frames per video
- Stored as `.npy` files in `MP_DATA/`  
📌 *Note: Folder too large to upload directly to GitHub — use local collection or compress as needed.*

---

## 🧠 Model Architecture
```python
LSTM(64) → LSTM(128) → LSTM(64) → Dense(64) → Dense(32) → Dense(5, softmax)
```

- Input shape: `(30, 1662)`  
- Output: One-hot encoded predictions for 5 gesture classes

---

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 📦 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/Saatvik2006/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Collect your own dataset using the data collection section in the script.

4. Load model or train your own:
   ```python
   model = load_model('action.h5')
   ```

5. Run real-time recognition:
   ```bash
   python your_main_script.py
   ```

---

## 📚 What I Learned
As an Electronics undergrad, this project taught me:
- How to use MediaPipe for pose/hand/face detection
- Real-time computer vision with OpenCV
- Structuring sequential data for LSTM training
- Building deep learning pipelines
- Debugging real-time models and visualizations
- Pushing large projects to GitHub (dealing with LFS limits)

---

## 🔧 Troubleshooting
- If you can't upload `MP_DATA`, zip and share externally or generate data locally.
- Webcam not detected? Ensure proper permissions or use `cv2.VideoCapture(1)` if using external cam.

---

## 📌 Credits
- Inspired by [Nicholas Renotte's tutorial]
- Extended with more gestures, UI polish, and structured into a full project

---

## 👤 Author
**Saatvik**  
Electronics and Communication Engineering Student  
[LinkedIn](www.linkedin.com/in/saatvik2706) • [GitHub](https://github.com/Saatvik2006)

---

⭐ If you like this project, consider giving it a star!
