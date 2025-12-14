# AI Sign Language to Speech 🤟 → 🔊

A **FREE**, **offline**, end-to-end AI project that converts sign language hand gestures into spoken words using computer vision and machine learning.

## 🎯 Project Overview

This project uses:
- **MediaPipe** for hand detection and landmark extraction
- **TensorFlow** for training a neural network to recognize gestures
- **OpenCV** for webcam access and visualization
- **pyttsx3** for offline text-to-speech

**No paid APIs. No internet required. Runs 100% locally on your computer.**

---

## 📁 Project Structure

```
ai-sign-to-speech/
│
├── config.py              # All configuration settings
├── utils.py               # Helper functions
├── hand_detector.py       # Hand detection using MediaPipe
├── collect_data.py        # Data collection script
├── train_model.py         # Model training script
├── predict_realtime.py    # Real-time prediction (testing)
├── text_builder.py        # Text formation logic
├── text_to_speech.py      # Offline TTS module
├── main.py                # Complete application
├── requirements.txt       # Python dependencies
│
├── data/
│   └── collected/         # CSV files with training data
│
└── models/                # Trained models saved here
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

**Note:** This may take a few minutes. All libraries are free and open-source.

---

### Step 2: Test Your Webcam and Hand Detection

Before collecting data, make sure hand detection works:

```bash
python hand_detector.py
```

- Show your hand to the webcam
- You should see green dots and lines on your hand
- Press 'q' to quit

**If this doesn't work, check:**
- Is your webcam connected?
- Is another program using the webcam?
- Are all dependencies installed?

---

### Step 3: Collect Training Data

Now collect training data for each letter (A-Z):

```bash
python collect_data.py
```

**What to do:**
1. The script will prompt you for each letter
2. Show the corresponding sign language gesture
3. Press 's' to start collecting samples
4. Move your hand slightly between samples (different angles, positions)
5. The script collects 100 samples per letter automatically
6. Take breaks between letters!

**Tips:**
- Use proper ASL (American Sign Language) hand shapes
- Vary your hand position and angle
- Keep your hand clearly visible
- Good lighting helps!

**Time estimate:** 15-30 minutes for all 26 letters

---

### Step 4: Train the Model

Once you have data for all letters, train the model:

```bash
python train_model.py
```

**What happens:**
1. Loads all your collected data
2. Builds a neural network
3. Trains the model (this may take 5-10 minutes)
4. Shows training progress and accuracy
5. Saves the trained model

**Look for:**
- Test accuracy > 85% is good
- Test accuracy > 95% is excellent
- If accuracy is low, collect more data

---

### Step 5: Test Real-Time Prediction

Test if the model recognizes your gestures:

```bash
python predict_realtime.py
```

- Show sign language gestures
- See predictions on screen
- Press 'q' to quit

**This is just for testing.** The full app is next!

---

### Step 6: Run the Complete Application

Run the full sign language to speech system:

```bash
python main.py
```

**How to use:**
1. Show sign language gestures to build words
2. Letters appear automatically when detected
3. Spaces are added when you pause (no hand visible)
4. Press **SPACE** to speak the current text
5. Press **BACKSPACE** to delete last character
6. Press **ESC** to clear all text
7. Press **H** to toggle help
8. Press **Q** to quit

---

## 🎮 Controls Summary

| Key | Action |
|-----|--------|
| **SPACE** | Speak the current text out loud |
| **BACKSPACE** | Delete the last character |
| **ESC** | Clear all text |
| **H** | Toggle help display |
| **Q** | Quit application |

---

## 🔧 Configuration

You can customize settings in `config.py`:

- **SAMPLES_PER_LETTER**: How many samples to collect (default: 100)
- **PREDICTION_CONFIDENCE_THRESHOLD**: Minimum confidence to accept predictions (default: 0.75)
- **STABILITY_FRAMES**: Frames needed for stable prediction (default: 5)
- **LETTER_REPEAT_DELAY**: Delay before repeating same letter (default: 1.5s)
- **SPACE_TRIGGER_DELAY**: Time with no hand to add space (default: 2.0s)
- **SPEECH_RATE**: Words per minute for TTS (default: 150)

---

## 🐛 Troubleshooting

### Webcam Issues

**Problem:** "Could not open webcam"

**Solutions:**
- Close other programs using the webcam (Zoom, Skype, etc.)
- Try changing `CAMERA_INDEX` in `config.py` (try 0, 1, 2)
- Check if webcam is properly connected

### Low Model Accuracy

**Problem:** Model accuracy < 70%

**Solutions:**
- Collect more training data (increase `SAMPLES_PER_LETTER`)
- Ensure good lighting when collecting data
- Use consistent hand shapes for each letter
- Vary hand positions during data collection
- Train for more epochs (increase `EPOCHS` in `config.py`)

### Hand Not Detected

**Problem:** "NO HAND DETECTED" even when showing hand

**Solutions:**
- Improve lighting
- Move hand closer to camera
- Ensure hand is fully visible
- Lower `MIN_DETECTION_CONFIDENCE` in `config.py`

### TTS Not Working

**Problem:** No voice output

**Solutions:**
- Check system volume
- Test TTS: `python text_to_speech.py`
- On Windows, ensure SAPI5 is available
- Try different voices (see `text_to_speech.py`)

### Flickering Predictions

**Problem:** Predictions change too quickly

**Solutions:**
- Increase `STABILITY_FRAMES` in `config.py`
- Increase `PREDICTION_CONFIDENCE_THRESHOLD`
- Hold gestures more steadily

---

## 📚 How It Works

### 1. Hand Detection (MediaPipe)
- Detects hand in webcam frame
- Finds 21 key points (landmarks) on the hand
- Normalizes coordinates relative to wrist

### 2. Data Collection
- Records landmark coordinates for each letter
- Saves to CSV files (one per letter)
- Each sample = 63 values (21 landmarks × 3 coordinates)

### 3. Model Training
- Loads all CSV data
- Builds neural network:
  - Input: 63 features
  - Hidden layer 1: 128 neurons + dropout
  - Hidden layer 2: 64 neurons + dropout
  - Output: 26 classes (A-Z)
- Trains using backpropagation
- Saves trained model

### 4. Real-Time Prediction
- Detects hand landmarks
- Feeds to trained model
- Gets probability for each letter
- Uses stability filter to prevent flickering

### 5. Text Formation
- Converts letters to words
- Prevents duplicate letters from same gesture
- Adds spaces when hand is not visible
- Manages sentence building

### 6. Text-to-Speech
- Converts text to speech using pyttsx3
- Works completely offline
- Adjustable speed and volume

---

## 🎯 Improving Accuracy

### Collect More Data
- More samples = better accuracy
- Aim for 200+ samples per letter
- Vary lighting, angles, distances

### Data Quality
- Use correct ASL hand shapes
- Keep hand clearly visible
- Consistent gesture for each letter
- Good lighting

### Model Tuning
- Increase epochs (more training time)
- Adjust learning rate
- Try different architectures
- Use data augmentation

---

## 🌟 Next Steps & Ideas

### Beginner Improvements
1. Add more letters or numbers
2. Add common words as single gestures
3. Improve UI with colors and animations
4. Add sound effects for feedback

### Intermediate Improvements
1. Add gesture for backspace/delete
2. Implement word suggestions
3. Save conversation history
4. Add multiple language support

### Advanced Improvements
1. Use LSTM for gesture sequences
2. Implement two-hand gestures
3. Add facial expressions for punctuation
4. Real-time translation to other languages
5. Mobile app version

---

## 📖 Learning Resources

### Sign Language
- [ASL Alphabet](https://www.startasl.com/asl-alphabet/)
- [Lifeprint ASL Dictionary](https://www.lifeprint.com/)

### Computer Vision
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

### Machine Learning
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Neural Networks Explained](https://www.youtube.com/watch?v=aircAruvnKk)

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Improve the code
- Add new features
- Fix bugs
- Enhance documentation

---

## 📝 License

This project is free and open-source. Use it for learning, teaching, or building upon!

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand detection
- **TensorFlow** for machine learning framework
- **OpenCV** for computer vision tools
- **pyttsx3** for offline text-to-speech

---

## ❓ FAQ

**Q: Do I need a GPU?**
A: No, this runs fine on CPU. Training takes 5-10 minutes on most computers.

**Q: Can I use this for other sign languages?**
A: Yes! Just collect data for your target sign language alphabet.

**Q: How accurate is it?**
A: With good training data, 90-95% accuracy is achievable.

**Q: Can I add words instead of just letters?**
A: Yes! Modify the code to add word-level gestures.

**Q: Does this work on Mac/Linux?**
A: Yes! All libraries are cross-platform.

---

## 📧 Need Help?

If you're stuck:
1. Check the Troubleshooting section above
2. Read the comments in the code files
3. Test each component individually
4. Make sure all dependencies are installed

---

**Happy coding! 🚀**
