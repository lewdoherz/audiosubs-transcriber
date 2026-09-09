import sys
import os
import site

# Automatically register pip-installed NVIDIA CUDA DLL directories for Windows
for site_pkg in site.getsitepackages():
    for subpath in [os.path.join("nvidia", "cublas", "bin"), os.path.join("nvidia", "cudnn", "bin")]:
        dll_path = os.path.join(site_pkg, subpath)
        if os.path.exists(dll_path):
            try:
                os.add_dll_directory(dll_path)
            except Exception:
                pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTextEdit, QLabel, QComboBox, QCheckBox,
    QTabWidget, QListWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from faster_whisper import WhisperModel

# App identity. APP_VERSION is bumped on every released iteration.
APP_NAME = "AudioSubs Transcriber"
APP_VERSION = "1.0.0"

# argostranslate provides fully offline machine translation. Language packs are
# downloaded once (small, a few dozen MB each) and cached for later offline use.
import argostranslate.package
import argostranslate.translate

# UI label -> argostranslate language code. Every non-None entry was verified
# against the live argosopentech package index (an en -> <code> model exists).
TRANSLATION_TARGETS = {
    "None": None,
    "Arabic": "ar",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Mandarin Chinese": "zh",
    "Norwegian": "nb",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
}

# Shared style for the read-only text panes (transcript + log).
CONSOLE_STYLE = """
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: Consolas, monospace;
    font-size: 13px;
    border-radius: 6px;
    padding: 10px;
"""

HELP_TEXT = f"""{APP_NAME} v{APP_VERSION}
Offline transcription + machine translation to .srt subtitles.

HOW TO USE
- Browse a single audio file (.mp3, .m4b, .wav) or a whole folder. Each audio file
  produces its own .srt subtitle file, saved next to the source.
- Pick a model: larger models are slower but more accurate (tiny ... large-v3).
- Device: 'auto' uses a CUDA GPU when available and falls back to CPU.
- Compute type: float16 is fastest on GPU; use int8 or float32 if you run out of memory.
- 'Translate to' adds a second .srt with the transcription machine-translated into the
  chosen language. The first run for a language downloads a small offline translation
  model (a few dozen MB, English -> language) and caches it for offline use.
- 'Also save combined' writes one file with original and translation together.

OUTPUT FILES
- name.srt            original transcription
- name.<lang>.srt     translation only (when a target language is chosen)
- name.en-<lang>.srt  original + translation combined (when enabled)

TABS
- Transcript: live list of completed subtitle cues (original and translation).
- Log: full operation log - model loading, downloads, saved files, errors.
- Output Files: paths of every .srt written during the current run.
"""


def ensure_translation_package_installed(from_code, to_code, log_callback=None):
    """Make sure the argostranslate package for from_code -> to_code is installed,
    downloading it if necessary. Safe to call every run (no-op if already installed)."""
    installed_languages = argostranslate.translate.get_installed_languages()
    have_from = any(lang.code == from_code for lang in installed_languages)
    have_to = any(lang.code == to_code for lang in installed_languages)

    if have_from and have_to:
        from_lang = next(l for l in installed_languages if l.code == from_code)
        to_lang = next(l for l in installed_languages if l.code == to_code)
        if from_lang.get_translation(to_lang) is not None:
            return

    if log_callback:
        log_callback(f"Downloading translation model ({from_code} -> {to_code})...")

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        (p for p in available_packages if p.from_code == from_code and p.to_code == to_code),
        None
    )
    if package_to_install is None:
        raise RuntimeError(f"No translation package available for {from_code} -> {to_code}")

    argostranslate.package.install_from_path(package_to_install.download())

    if log_callback:
        log_callback(f"Translation model ({from_code} -> {to_code}) ready.")


def get_translation(from_code, to_code):
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next(l for l in installed_languages if l.code == from_code)
    to_lang = next(l for l in installed_languages if l.code == to_code)
    return from_lang.get_translation(to_lang)


class TranscriptionWorker(QThread):
    progress_signal = pyqtSignal(str)          # log stream messages
    banner_signal = pyqtSignal(str)            # per-file headers for the transcript tab
    segment_signal = pyqtSignal(str, str, str, str)  # start, end, original, translation
    output_signal = pyqtSignal(str)            # absolute path of each saved .srt
    finished_signal = pyqtSignal(str)

    def __init__(self, audio_paths, model_size, device_mode, compute_type, translate_to_code=None, combine_bilingual=False):
        super().__init__()
        self.audio_paths = audio_paths if isinstance(audio_paths, list) else [audio_paths]
        self.model_size = model_size
        self.device_mode = device_mode
        self.compute_type = compute_type
        self.translate_to_code = translate_to_code  # e.g. "zh", "es", or None
        self.combine_bilingual = combine_bilingual  # also write a combined original+translation SRT

    def run(self):
        try:
            self.progress_signal.emit(f"Loading Whisper model ('{self.model_size}') on [{self.device_mode.upper()}] with compute type [{self.compute_type}]...")

            model = WhisperModel(
                self.model_size,
                device=self.device_mode,
                compute_type=self.compute_type
            )

            translator = None
            if self.translate_to_code:
                ensure_translation_package_installed(
                    "en", self.translate_to_code, log_callback=self.progress_signal.emit
                )
                translator = get_translation("en", self.translate_to_code)

            total_files = len(self.audio_paths)
            for index, audio_path in enumerate(self.audio_paths, start=1):
                file_name = os.path.basename(audio_path)
                self.progress_signal.emit(f"\n[{index}/{total_files}] Processing: {file_name}")
                self.progress_signal.emit("-" * 40)
                self.banner_signal.emit(f"[{index}/{total_files}] {file_name}")

                segments, info = model.transcribe(audio_path, beam_size=5)

                # Materialize the generator once so we can write the original SRT
                # and, if requested, translate each line for a second SRT.
                collected_segments = []
                for segment in segments:
                    collected_segments.append((segment.start, segment.end, segment.text.strip()))

                base_name = os.path.splitext(audio_path)[0]
                srt_path = f"{base_name}.srt"

                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, (start, end, text) in enumerate(collected_segments, start=1):
                        start_str = self.format_time(start)
                        end_str = self.format_time(end)

                        f.write(f"{i}\n")
                        f.write(f"{start_str} --> {end_str}\n")
                        f.write(f"{text}\n\n")

                        self.progress_signal.emit(f"[{start_str} --> {end_str}] {text}")
                        if translator is None:
                            self.segment_signal.emit(start_str, end_str, text, "")

                self.progress_signal.emit(f"Saved: {os.path.basename(srt_path)}")
                self.output_signal.emit(srt_path)

                if translator is not None:
                    translated_path = f"{base_name}.{self.translate_to_code}.srt"
                    self.progress_signal.emit(f"Translating to '{self.translate_to_code}'...")

                    # Translate once per segment, reused for both the translated-only
                    # file and the combined bilingual file so we don't call the
                    # translator twice per line.
                    translated_texts = []
                    for start, end, text in collected_segments:
                        translated_text = translator.translate(text) if text else ""
                        translated_texts.append(translated_text)
                        start_str = self.format_time(start)
                        end_str = self.format_time(end)
                        self.progress_signal.emit(f"[{start_str} --> {end_str}] {translated_text}")
                        self.segment_signal.emit(start_str, end_str, text, translated_text)

                    with open(translated_path, "w", encoding="utf-8") as f:
                        for i, ((start, end, _text), translated_text) in enumerate(
                            zip(collected_segments, translated_texts), start=1
                        ):
                            start_str = self.format_time(start)
                            end_str = self.format_time(end)
                            f.write(f"{i}\n")
                            f.write(f"{start_str} --> {end_str}\n")
                            f.write(f"{translated_text}\n\n")

                    self.progress_signal.emit(f"Saved: {os.path.basename(translated_path)}")
                    self.output_signal.emit(translated_path)

                    if self.combine_bilingual:
                        combined_path = f"{base_name}.en-{self.translate_to_code}.srt"
                        with open(combined_path, "w", encoding="utf-8") as f:
                            for i, ((start, end, text), translated_text) in enumerate(
                                zip(collected_segments, translated_texts), start=1
                            ):
                                start_str = self.format_time(start)
                                end_str = self.format_time(end)
                                f.write(f"{i}\n")
                                f.write(f"{start_str} --> {end_str}\n")
                                f.write(f"{text}\n")
                                f.write(f"{translated_text}\n\n")

                        self.progress_signal.emit(f"Saved: {os.path.basename(combined_path)}")
                        self.output_signal.emit(combined_path)

            self.finished_signal.emit(f"\n{'*'*40}\nAll {total_files} file(s) transcribed successfully!")
        except Exception as e:
            self.finished_signal.emit(f"Error during transcription: {str(e)}")

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

class TranscribeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(800, 600)

        self.audio_paths = []
        self.worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("No audio file or folder selected.")
        self.file_label.setStyleSheet("color: #a0a0a0; font-style: italic;")

        self.browse_file_btn = QPushButton("Browse File")
        self.browse_file_btn.clicked.connect(self.select_single_file)

        self.browse_folder_btn = QPushButton("Browse Folder")
        self.browse_folder_btn.clicked.connect(self.select_folder)

        file_layout.addWidget(self.file_label, stretch=1)
        file_layout.addWidget(self.browse_file_btn)
        file_layout.addWidget(self.browse_folder_btn)
        layout.addLayout(file_layout)

        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("medium")
        settings_layout.addWidget(self.model_combo)

        settings_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "auto"])
        self.device_combo.setCurrentText("auto")
        settings_layout.addWidget(self.device_combo)

        settings_layout.addWidget(QLabel("Compute Type:"))
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["float16", "int8", "float32"])
        self.compute_combo.setCurrentText("float16")
        settings_layout.addWidget(self.compute_combo)

        layout.addLayout(settings_layout)

        translate_layout = QHBoxLayout()
        translate_layout.addWidget(QLabel("Translate to:"))
        self.translate_combo = QComboBox()
        self.translate_combo.addItems(list(TRANSLATION_TARGETS.keys()))
        self.translate_combo.setCurrentText("None")
        self.translate_combo.currentTextChanged.connect(self.update_combine_checkbox_enabled)
        translate_layout.addWidget(self.translate_combo)

        self.combine_checkbox = QCheckBox("Also save combined (original + translation) file")
        self.combine_checkbox.setEnabled(False)
        translate_layout.addWidget(self.combine_checkbox)

        translate_layout.addStretch(1)
        layout.addLayout(translate_layout)

        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2d5a27; color: white;")
        self.start_btn.clicked.connect(self.start_transcription)
        layout.addWidget(self.start_btn)

        self.tabs = QTabWidget()

        # Tab 1: live transcript of completed subtitle cues (original + translation).
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setStyleSheet(CONSOLE_STYLE)
        self.transcript_display.setText("Transcribed (and, if selected, translated) subtitle cues will appear here as each segment is completed.")
        self.tabs.addTab(self.transcript_display, "Transcript")

        # Tab 2: full operation log (model loading, downloads, saves, errors).
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(CONSOLE_STYLE)
        self.log_display.setText("Select files or a folder, choose your hardware settings, and click 'Start Transcription'.")
        self.tabs.addTab(self.log_display, "Log")

        # Tab 3: every .srt written during the current run.
        self.output_list = QListWidget()
        self.output_list.setStyleSheet("""
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: Consolas, monospace;
            font-size: 12px;
            border-radius: 6px;
            padding: 6px;
        """)
        self.tabs.addTab(self.output_list, "Output Files")

        # Tab 4: usage reference.
        self.help_display = QTextEdit()
        self.help_display.setReadOnly(True)
        self.help_display.setText(HELP_TEXT)
        self.tabs.addTab(self.help_display, "Help")

        layout.addWidget(self.tabs)

    def update_combine_checkbox_enabled(self, translate_choice_text):
        translate_enabled = TRANSLATION_TARGETS[translate_choice_text] is not None
        self.combine_checkbox.setEnabled(translate_enabled)
        if not translate_enabled:
            self.combine_checkbox.setChecked(False)

    def select_single_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.mp3 *.m4b *.wav)")
        if file_name:
            self.audio_paths = [file_name]
            self.file_label.setText(os.path.basename(file_name))
            self.file_label.setStyleSheet("color: #ffffff; font-weight: bold;")
            self.start_btn.setEnabled(True)
            self.log_display.append(f"Selected file: {file_name}")

    def select_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Audiobooks")
        if dir_path:
            supported_extensions = (".mp3", ".m4b", ".wav")
            found_files = [
                os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith(supported_extensions)
            ]

            if found_files:
                self.audio_paths = sorted(found_files)
                self.file_label.setText(f"Folder: {os.path.basename(dir_path)} ({len(self.audio_paths)} audio files)")
                self.file_label.setStyleSheet("color: #ffffff; font-weight: bold;")
                self.start_btn.setEnabled(True)
                self.log_display.append(f"Found {len(self.audio_paths)} audio files in folder: {dir_path}")
            else:
                self.file_label.setText("No audio files found in folder.")
                self.start_btn.setEnabled(False)

    def start_transcription(self):
        if not self.audio_paths:
            return

        self.browse_file_btn.setEnabled(False)
        self.browse_folder_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.compute_combo.setEnabled(False)
        self.translate_combo.setEnabled(False)
        self.combine_checkbox.setEnabled(False)
        self.log_display.clear()
        self.transcript_display.clear()
        self.output_list.clear()

        model_size = self.model_combo.currentText()
        device_mode = self.device_combo.currentText()
        compute_type = self.compute_combo.currentText()
        translate_to_code = TRANSLATION_TARGETS[self.translate_combo.currentText()]
        combine_bilingual = self.combine_checkbox.isChecked()

        self.worker = TranscriptionWorker(
            self.audio_paths, model_size, device_mode, compute_type, translate_to_code, combine_bilingual
        )
        self.worker.progress_signal.connect(self.log_message)
        self.worker.banner_signal.connect(self.append_transcript_banner)
        self.worker.segment_signal.connect(self.append_transcript_segment)
        self.worker.output_signal.connect(self.add_output_file)
        self.worker.finished_signal.connect(self.transcription_finished)
        self.worker.start()

    def log_message(self, message):
        self.log_display.append(message)

    def append_transcript_banner(self, banner):
        self.transcript_display.append(f"{'=' * 40}\n{banner}\n{'=' * 40}")

    def append_transcript_segment(self, start_str, end_str, text, translation):
        lines = f"{start_str} --> {end_str}\n{text}"
        if translation:
            lines += f"\n{translation}"
        self.transcript_display.append(lines)

    def add_output_file(self, path):
        if self.output_list.findItems(path, Qt.MatchFlag.MatchExactly):
            return
        self.output_list.addItem(path)

    def transcription_finished(self, message):
        self.log_display.append("\n" + message)
        self.transcript_display.append("\n" + message)
        self.browse_file_btn.setEnabled(True)
        self.browse_folder_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.compute_combo.setEnabled(True)
        self.translate_combo.setEnabled(True)
        self.update_combine_checkbox_enabled(self.translate_combo.currentText())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranscribeWindow()
    window.show()
    sys.exit(app.exec())