# Signature Verification System  
![CI Status](https://github.com/sridhar1pro/signature-verification/actions/workflows/python-app.yml/badge.svg)

Offline signature verification implemented in Python using OpenCV + Dynamic Programming (DTW) + greedy matching.

---

## Features

- Preprocessing and skeletonization (scikit-image)
- Contour → resampled feature sequences
- Greedy matching (fast filter) + DTW (refined)
- CLI demo (`src/demo.py`) and simple GUI (`gui.py`)
- Debug visualizations saved into `data/debug/`
- Greedy + DTW based signature verification
- GUI for interactive use
- Debug visualization automatically saved

---

## Quick Start (Windows PowerShell)

```powershell
cd your project path\signature_verification
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run CLI
python src/demo.py

# OR run GUI
python gui.py
```
