# Palm-Recognition-using-LBPH


This is a simple command-line application for palm-print–based user registration and login using OpenCV’s Local Binary Patterns Histograms (LBPH) recognizer.

## Features

* **Register** new users by capturing palm images from a webcam
* **Train** an LBPH model on all registered palm samples
* **Login** by recognizing a live palm and returning the user’s name
* **Pure Python** with one dependency: `opencv-contrib-python` and `numpy`

## Repository Structure

```
├── palm_recognition.py     # Main application script
├── palm_dataset/           # Captured palm images (auto-generated)
│   └── <Name>_#.jpg        # Cropped grayscale palm samples
├── palm_labels.pickle      # Persisted mapping: name → numeric ID
├── palm_trainer.yml        # Trained LBPH model data
└── PALM_README.md          # This documentation
```

## Prerequisites

* Python 3.7+
* Webcam or USB camera

## Installation

1. Copy `palm_recognition.py` into your project folder.
2. Install dependencies:

   ```bash
   pip install opencv-contrib-python numpy
   ```

## Usage

Run the script and choose between **register** and **login** modes:

```bash
python palm_recognition.py
```

### 1. Register a New User

1. At the prompt, enter `register`.
2. Enter the user’s name (e.g. `Bob`).
3. A webcam window opens showing a centered square box. Place your palm inside.
4. Press **c** to capture each sample (default 20). Console logs show progress.
5. Press **q** at any time to abort.
6. After capturing, the LBPH model retrains automatically and saves `palm_trainer.yml` and `palm_labels.pickle`.

### 2. Login (Recognize a Palm)

1. At the prompt, enter `login`.
2. A webcam window opens with the same box overlay.
3. Place your palm inside; the script computes LBPH features and predicts a label.
4. If the confidence is below the threshold, your name appears; otherwise, `Unknown`.
5. Press **q** or hold still until a known palm is detected to exit.

## How It Works

1. **ROI Detection**: Uses a fixed centered box to standardize the palm region.
2. **Feature Extraction (LBPH)**: Computes histograms of local binary patterns on the palm crop.
3. **Training**: Stores histograms and labels in `palm_trainer.yml`.
4. **Prediction**: Compares a live histogram against stored examples to find the closest match.

## Configuration

* **Sample Count**: Modify `SAMPLE_COUNT` at the top of `palm_recognition.py` to change the number of registration captures.
* **Threshold**: Adjust the `threshold` parameter in `recognize_user()` to fine-tune recognition sensitivity.

## License

MIT License. Feel free to use, modify, or distribute.

## Acknowledgments

* Based on OpenCV’s LBPH face recognizer, adapted for palm prints.
* Utilizes OpenCV Haar cascades for region-of-interest standardization.
