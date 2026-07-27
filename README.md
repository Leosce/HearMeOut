
[video]https://github.com/user-attachments/assets/1f13a3aa-0159-4aff-9641-237b49eaaee9

# HearMeOut

**HearMeOut** turns silent lip movement into text and speech. Point a camera at someone talking — live or from an uploaded clip — and get a real-time transcription, with optional Arabic translation and text-to-speech playback. It's built as an accessibility tool for lip-reading assistance, aimed at deaf and hard-of-hearing users and noisy/silent environments where audio isn't available.


## How it works

A visual-speech-recognition (VSR) model looks at the mouth region across a sequence of frames and predicts the words being spoken — no audio required. HearMeOut wraps this in three purpose-built "readers":

| Reader | Task | Notes |
| --- | --- | --- |
| **Aurora** | Full-sentence English reading | Live recordings and uploaded video |
| **Esma3ny** | Arabic word / short-phrase reading | Uploaded clips, silence-based phrase splitting |
| **Lyra** | Compact reader for short, front-facing English words | Uploaded video only |

Predicted English text can be translated to Arabic and read back with text-to-speech.

## Clients

HearMeOut ships as several front ends, all talking to the same backend:

- **`android_app/`** — native Android client (Java)
- **`flutter_app/`** — Flutter Android client
- **`desktop_launcher/`** — Windows desktop launcher (C#) that boots the backend and opens the web UI in an app window
- **`hear-me-out-v3.html`** — the web UI itself, served by the backend
- **`mobile_app/`** — the backend: a dependency-free Python `http.server` app with SQLite-backed accounts/history, serving the web UI to phones over the local network and running model inference

## Repository layout

```
HearMeOut/
├── android_app/          Native Android client
├── flutter_app/           Flutter Android client
├── desktop_launcher/       Windows desktop launcher (C#)
├── mobile_app/            Backend server (Python, stdlib http.server + SQLite)
├── avhubert_ext/           Custom fairseq task/dataset extensions for joint fine-tuning
├── scripts/                Inference, dataset-prep, and training pipeline scripts
├── lyra_files/             Lyra (GRID reader) model definition and preprocessing
├── conf/                  Fine-tuning configs
├── assets/                 App icon
├── reference_screens/       UI reference screenshots
├── hear-me-out-v3.html      Web UI
└── build_desktop_exe.ps1    Builds the desktop launcher exe
```

## Setup

The VSR engine builds on [AV-HuBERT](https://facebookresearch.github.io/av_hubert/) (Meta AI Research) and [fairseq](https://github.com/facebookresearch/fairseq). Their source and pretrained checkpoints are **not bundled in this repository** — AV-HuBERT is distributed under a **non-commercial license (CC BY-NC 4.0)**, so this repo only contains original code that *targets* that framework, not the framework itself. To run inference locally:

1. Clone the upstream projects and follow their setup instructions:
   ```powershell
   git clone https://github.com/facebookresearch/fairseq.git
   git clone https://github.com/facebookresearch/av_hubert.git
   ```
2. Obtain a pretrained checkpoint from the [official AV-HuBERT release page](https://facebookresearch.github.io/av_hubert/) and review its license terms before use.
3. Point the scripts in `scripts/` at your local `fairseq`/`av_hubert` checkouts and checkpoint path.
4. Start the backend:
   ```powershell
   cd mobile_app
   python server.py
   ```
   This prints a local-network URL that the Android/Flutter/desktop clients connect to.

`avhubert_ext/` contains this project's own fairseq task and dataset extensions (joint fine-tuning support) — that part *is* original code and is included.

## Tech stack

- **VSR model**: AV-HuBERT (audio-visual speech representation learning) fine-tuned for this project's readers
- **Face/mouth ROI**: dlib / face-alignment landmark detection
- **Translation**: `Helsinki-NLP/opus-mt-en-ar` (MarianMT)
- **Backend**: Python standard library `http.server`, SQLite
- **Clients**: Java (Android), Flutter/Dart, C# (Windows desktop launcher), HTML/CSS/JS (web UI)

## License

Original code in this repository is licensed under the [MIT License](LICENSE) © Leosce.

This project depends on, but does not redistribute, AV-HuBERT and fairseq (Meta AI Research), which are licensed separately (AV-HuBERT: CC BY-NC 4.0 — non-commercial use only). If you use this project, make sure your use of those components complies with their own license terms.

## Acknowledgments

- Shi, Bowen, et al. ["Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction."](https://arxiv.org/abs/2201.02184) ICLR 2022. (AV-HuBERT)
- [fairseq](https://github.com/facebookresearch/fairseq) — Facebook AI Research Sequence-to-Sequence Toolkit
- [Helsinki-NLP/opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) — English–Arabic translation
