# AudioSubs Transcriber

Offline audiobook/audio transcription to `.srt` subtitles with optional machine
translation, backed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for speech-to-text and [Argos Translate](https://github.com/argosopentech/argos-translate)
for fully offline translation.

## Features

- Transcribe single audio files (`.mp3`, `.m4b`, `.wav`) or an entire folder of them
- Model selection: `tiny` (fastest) through `large-v3` (most accurate)
- CUDA GPU or CPU device modes with selectable compute types (`float16`, `int8`, `float32`)
- Translate to 28 languages, all offline after a one-time model download
- Live Transcript tab showing completed subtitle cues as they are produced
- Separate Log, Output Files, and Help tabs
- Optional combined bilingual (original + translation) subtitle file

## Requirements

- Python 3.10+ (developed on 3.14)
- NVIDIA GPU + CUDA for `cuda` device mode (falls back to CPU with `auto`)

## Install from source

```bash
pip install -r requirements.txt   # PyQt6, faster-whisper, argostranslate
python transcribe_ui.py
```

## Usage

1. **Browse File** or **Browse Folder** to pick the audio source.
2. Choose model, device, and compute type.
3. Optionally pick a **Translate to** language; first use for a language downloads a
   small offline translation model (a few dozen MB), cached for later runs.
4. Click **Start Transcription**.

Output `.srt` files are written next to each source audio file:

| File | Contents |
| --- | --- |
| `name.srt` | original transcription |
| `name.<lang>.srt` | translation only (when a target language is selected) |
| `name.en-<lang>.srt` | original + translation combined (when enabled) |

## Tabs

- **Transcript** - live list of completed cues (original and translation)
- **Log** - full operation log: model loading, downloads, saved files, errors
- **Output Files** - paths of every `.srt` written during the current run
- **Help** - usage reference

## Build the executable

```bash
pyinstaller transcribe_ui.spec --noconfirm
```

Produces `dist/AudioSubs Transcriber.exe`. Versioned binaries are published as
[GitHub Releases](https://github.com/lewdoherz/audiosubs-transcriber/releases).

## Versioning

`APP_VERSION` in `transcribe_ui.py` is the current iteration version and is
displayed in the window title and Help tab. Bump it for each released iteration
and tag the commit (`v<APP_VERSION>`) when publishing a Release.
