# HearMeOut Android App

This is a native Android client for the laptop-hosted HearMeOut backend. It is
not a localhost web page: install it on the Android phone, enter the laptop
server URL, and the app sends recordings/uploads to the laptop for prediction.

## What It Does

- Start/instruction screens.
- Sign up and login against the laptop SQLite database.
- Success screen after signup/login.
- Home page with Record Live, Upload Video, History, Settings, and accuracy tips.
- Record Live opens the Android camera app, captures a clip, uploads it to the
  laptop, polls for the model result, and shows the prediction.
- Upload Video opens the Android file picker, uploads the selected video, and
  shows the prediction.
- History reads previous predictions from the laptop database.
- Translation calls the laptop Arabic translation endpoint.
- Text-to-speech uses Android's built-in TTS engine on the phone.

## Build Requirements

This workspace currently does not have Flutter, Dart, Gradle, or Android Studio
available on PATH, so I could not build an APK here. To build it:

1. Install Android Studio on the laptop.
2. Open `D:\HearMeOut\HearMeOut\android_app` in Android Studio.
3. Let Android Studio sync Gradle.
4. Connect the Android phone with USB debugging enabled, or create an APK from
   `Build > Build Bundle(s) / APK(s) > Build APK(s)`.

The project uses plain Java and Android SDK APIs, so there are no extra app
dependencies.

## Run With The Laptop Server

Start the laptop backend first:

```powershell
cd D:\HearMeOut\HearMeOut
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_mobile_app.ps1
```

The backend prints a phone URL like:

```text
http://192.168.1.20:8000
```

Open the Android app, paste that URL into the server field on Sign Up, Login,
or Settings, then sign up or log in.

## Quick Backend Test

To test the app flow without running AV-HuBERT:

```powershell
.\run_mobile_app.ps1 -Mock
```

Mock mode returns `Hello, how are you?` for every prediction.

## Important Network Note

The Android app should not use `localhost` for the laptop. On Android,
`localhost` means the phone itself. Use the laptop Wi-Fi IP printed by
`run_mobile_app.ps1`.
