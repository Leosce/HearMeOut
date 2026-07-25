# HearMeOut Flutter Android App

This is the native Android Flutter client for HearMeOut. The phone app captures
or uploads video, then sends it to the laptop backend. The laptop keeps the
SQLite users/history database and runs the AV-HuBERT model.

## Built APK

The current release APK is here:

```text
D:\HearMeOut\HearMeOut\flutter_app\build\app\outputs\flutter-apk\app-release.apk
```

## Start The Laptop Backend

In PowerShell:

```powershell
cd D:\HearMeOut\HearMeOut
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_mobile_app.ps1
```

PowerShell prints URLs like:

```text
[run] app:    http://localhost:8000
[run] phone:  http://192.168.1.20:8000
```

Use the `phone` URL inside the Android app. Do not use `localhost` on a real
phone, because that points to the phone itself, not the laptop.

## Install On Android

1. Put the phone and laptop on the same Wi-Fi network.
2. Copy/install `app-release.apk` on the Android phone.
3. If Android blocks it, allow installing from this source.
4. Open HearMeOut.
5. On Sign Up or Login, enter the laptop server URL printed by PowerShell.
6. Create an account or log in.

## What Works

- Start/instruction screens.
- Sign up and login using the laptop SQLite database.
- Success screen after authentication.
- Home, History, and Settings bottom navigation.
- Live recording through Android camera capture.
- Prerecorded video upload through Android gallery/file picker.
- Laptop prediction polling with progress and result screen.
- History saved and loaded from the laptop.
- English to Arabic translation through the laptop backend.
- Text-to-speech on the phone through Android TTS.
- Editable laptop server URL and model detector settings.

## Build Again

Flutter is configured to use:

```text
Android SDK: D:\HearMeOut\AndroidSdk
JDK:         C:\Program Files\Java\jdk-21
```

Build:

```powershell
cd D:\HearMeOut\HearMeOut\flutter_app
flutter build apk --release
```

If the Android SDK is ever missing on a fresh machine, run:

```powershell
cd D:\HearMeOut\HearMeOut
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\install_android_sdk_for_flutter.ps1
```

## Verification Run

These passed after the APK build:

```powershell
flutter analyze
flutter test
flutter doctor -v
flutter build apk --release
```
