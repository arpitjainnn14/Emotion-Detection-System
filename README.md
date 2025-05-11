# Real-Time Face and Mood Detection

This project implements a real-time face and mood detection system using computer vision and deep learning. It can detect faces in a video stream and analyze the emotional state of detected faces.

## Features

- Real-time face detection
- Emotion analysis (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)
- Real-time emotion visualization
- Emotion statistics tracking
- Screenshot capture
- Emotion logging for analysis
- User-friendly GUI interface

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

- Press 'q' to quit the application
- Press 's' to capture a screenshot
- The application will automatically log emotions to a CSV file

## Project Structure

- `main.py`: Main application entry point
- `face_detector.py`: Face detection module
- `emotion_analyzer.py`: Emotion analysis module
- `gui.py`: GUI interface
- `utils.py`: Utility functions
- `logs/`: Directory for emotion logs
- `screenshots/`: Directory for captured screenshots

## License

MIT License 