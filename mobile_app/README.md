# HearMeOut Phone App

This is a phone-friendly web app served by the laptop. The Android phone opens
the laptop URL in a browser, while the laptop keeps the SQLite database and runs
the AV-HuBERT lip-reading model.

## Start The Laptop Server

From PowerShell in `D:\HearMeOut\HearMeOut`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_mobile_app.ps1
```

The script prints URLs like:

```text
[run] app:    http://localhost:8000
[run] phone:  http://192.168.1.20:8000
```

Open the `http://192.168...:8000` URL on the Android phone while the phone and
laptop are on the same Wi-Fi network.

Important: on the phone, `localhost` means the phone itself. Use the laptop IP
address printed by the script.

## Quick UI Test Without Running The Model

```powershell
.\run_mobile_app.ps1 -Mock
```

Mock mode returns `Hello, how are you?` for predictions so you can test signup,
login, history, translation buttons, and the phone layout quickly.

## Data Stored On The Laptop

- Users, sessions, settings, and prediction history:
  `mobile_app\data\hearmeout.db`
- Uploaded videos:
  `mobile_app\uploads`
- Model output folders:
  `runs\mobile_app`

## Android Camera Note

Android browsers only allow live camera preview through a secure browser
context. The app tries live preview first. If the browser blocks it on the Wi-Fi
HTTP URL, use the camera picker shown on the Live Recording page. It records a
clip on the phone and uploads it to the laptop for prediction.

Upload Video always works through the normal Android file picker.

## Model Notes

- The laptop runs `scripts\decode_video.py` using the same AV-HuBERT setup as
  the PowerShell decoders.
- Confidence scores are shown only if a decoder exposes them. The current
  AV-HuBERT path returns text but not calibrated confidence.
- Arabic translation is run on the laptop because Marian translation is not
  lightweight enough to bundle into this browser app. Text-to-speech uses the
  Android browser's built-in speech engine.
