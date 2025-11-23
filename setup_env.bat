@echo off
echo Creating virtual environment...
py -3.10 -m venv venv
call venv\Scripts\activate

echo Installing compatible packages...
pip install --upgrade pip
pip install numpy==1.26.4
pip install scipy==1.10.1
pip install scikit-image==0.21.0
pip install opencv-python==4.7.0.72
pip install pillow==12.0.0

echo Done. Activate with:
echo     venv\Scripts\Activate.ps1
pause
