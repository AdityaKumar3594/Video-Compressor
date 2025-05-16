# Video Compression using K-Means Clustering and PCA

This project demonstrates video compression using two fundamental data mining techniques:
1. K-Means Clustering for color reduction
2. Principal Component Analysis (PCA) for dimensionality reduction

## Project Structure
- `main.py`: Main script for video compression
- `utils.py`: Helper functions for video processing
- `requirements.txt`: Project dependencies

## Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your input video in the project directory
2. Run the main script:
```bash
python main.py --input video.mp4 --output compressed_video.mp4
```

## Features
- Frame extraction and preprocessing
- Color reduction using K-Means clustering
- Dimensionality reduction using PCA
- Video reconstruction and quality metrics
- Visualization of original vs compressed frames

## Evaluation Metrics
- Compression Ratio
- Peak Signal-to-Noise Ratio (PSNR)
- Visual quality comparison

## License
MIT License 