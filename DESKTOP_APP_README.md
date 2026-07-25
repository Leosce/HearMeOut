# HearMeOut Desktop App

## Start In Production Mode

Double-click:

```powershell
D:\HearMeOut\HearMeOut\HearMeOutDesktop.exe
```

Or run from PowerShell:

```powershell
cd D:\HearMeOut\HearMeOut
.\HearMeOutDesktop.exe
```

Production mode is the default. It starts the laptop backend and opens `hear-me-out-v3.html` in a browser app window.

## Start Without Browser

Useful for testing the backend only:

```powershell
cd D:\HearMeOut\HearMeOut
.\HearMeOutDesktop.exe -NoBrowser
```

## Start In Simulation Mode

Simulation mode returns mock predictions. Use this only for UI demos:

```powershell
cd D:\HearMeOut\HearMeOut
.\HearMeOutDesktop.exe -Simulation
```

The equivalent PowerShell command is:

```powershell
.\run_hearmeout_desktop.ps1 -Simulation
```

## Stop The Backend

```powershell
cd D:\HearMeOut\HearMeOut
.\stop_hearmeout_desktop.ps1
```

## Models

- Aurora Sentence Reader: AV-HuBERT sentence reader. Works for live recordings and uploaded videos.
- Esma3ny: Arabic word and short-phrase reader for uploaded clips. The app uses short pauses to separate words, then joins the predicted words into one result.
- Lyra GRID Reader: Compact silent-speech reader for short, front-facing English GRID clips. Available for uploaded videos only.

Live recording in the desktop UI records a short video clip first, then sends that clip to the backend model. Live mode currently offers Aurora only; Esma3ny and Lyra are available for uploaded videos.

## Esma3ny Phrase Splitting

Esma3ny uses the audio track only to find word boundaries. By default, a silence of about `0.5s` closes the current word segment. Each segment is then sent to the same visual LRW-AR classifier. Very long detected regions are capped at about `2.2s` so noisy audio does not create a huge word segment.

Direct script example:

```powershell
cd D:\HearMeOut\HearMeOut
E:\Anaconda\envs\avhubert\python.exe .\scripts\decode_lrw_ar.py --video .\your_video.mp4 --phrase-mode --min-silence-ms 500 --audio-threshold-db auto
```

For clearer phrase results, speak one supported Esma3ny vocabulary word at a time and leave a short pause between words.

Esma3ny preprocessing is matched to the training notebook:

- dlib 68-point detector/predictor.
- mouth landmarks `48..67`.
- fixed `15px` mouth padding.
- grayscale `96x96` mouth crops using OpenCV default resize interpolation.
- center trim or last-frame pad to `29` frames.
- per-clip mean/std normalization.
- tensor shape `(1, 1, 29, 96, 96)`.

## Translation And TTS

- Arabic translation runs on the laptop backend through the cached `Helsinki-NLP/opus-mt-en-ar` model.
- Desktop result speech uses the browser `speechSynthesis` engine. Select `Original` or `Arabic` on the result screen before pressing the speaker button.
- If a model already returns Arabic text, the app does not translate it again.

## Rebuild The EXE

If you edit the launcher source:

```powershell
cd D:\HearMeOut\HearMeOut
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_desktop_exe.ps1
```
