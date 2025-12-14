# 🚀 Quick Start Commands - Keep This Open!

## Every Time You Start Working

### 1️⃣ Open PowerShell
Search "PowerShell" in Windows Start menu

### 2️⃣ Go to Project Folder
```powershell
cd "C:\Users\Orosmit Mishra\Desktop\ai-sign-to-speech"
```

### 3️⃣ Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
✅ You should see `(venv)` at the start of your line

### 4️⃣ Run Your Code
```powershell
python hand_detector.py
```

---

## When You're Done Working

```powershell
deactivate
```

---

## First Time Setup (Do Once)

1. Download Python 3.11 from https://www.python.org/downloads/
2. Install it (check "Add Python to PATH"!)
3. Open PowerShell and run:

```powershell
cd "C:\Users\Orosmit Mishra\Desktop\ai-sign-to-speech"
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Common Issues

**"Activate.ps1 cannot be loaded"**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"No module named 'mediapipe'"**
- Make sure you see `(venv)` in your PowerShell
- If not, run: `.\venv\Scripts\Activate.ps1`

**Webcam not working**
- Close other apps using webcam
- Press 'q' to quit the program

---

## Check If Everything Works

```powershell
python -c "import mediapipe, cv2, tensorflow; print('✅ Ready to go!')"
```

Should show: `✅ Ready to go!`
