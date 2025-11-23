# Signature Verification System

Offline signature verification implemented in Python using OpenCV + Dynamic Programming (DTW) + greedy matching.

## Features

- Preprocessing and skeletonization (scikit-image)
- Contour → resampled feature sequences
- Greedy matching (fast filter) + DTW (refined)
- CLI demo (`src/demo.py`) and simple GUI (`gui.py`)
- Debug visualizations saved into `data/debug/`

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\siban\signature_verification
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/demo.py
# or
python gui.py
## Features
- Greedy + DTW based signature verification
- Uses OpenCV preprocessing + skeletonization
- GUI for interactive use
- Debug visualization automatically saved

## Demo
<img width="1220" height="689" alt="Screenshot (31)" src="https://github.com/user-attachments/assets/4aa8914f-5d44-45f8-9519-273c8b0be503" />
