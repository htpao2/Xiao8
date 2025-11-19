import subprocess
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes using the compiled whisper.cpp executable.

    Args:
        audio_bytes: The raw audio data in bytes.

    Returns:
        The transcribed text as a string.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_filepath = temp_audio_file.name

    try:
        whisper_executable = "./whisper.cpp/bin/whisper-cli"
        model_path = "./whisper.cpp/models/ggml-tiny.bin"

        if not os.path.exists(whisper_executable):
            raise FileNotFoundError("whisper.cpp executable not found.")
        if not os.path.exists(model_path):
            raise FileNotFoundError("whisper.cpp model not found.")

        command = [
            whisper_executable,
            "-m", model_path,
            "-f", temp_audio_filepath,
            "-otxt",  # Output in plain text
        ]

        logger.info("Running whisper.cpp command...")
        subprocess.run(command, check=True, capture_output=True, text=True)

        output_txt_path = temp_audio_filepath + ".txt"
        if not os.path.exists(output_txt_path):
            raise FileNotFoundError("Whisper.cpp output file not found.")

        with open(output_txt_path, "r", encoding="utf-8") as f:
            transcribed_text = f.read().strip()

        return transcribed_text

    finally:
        if os.path.exists(temp_audio_filepath):
            os.remove(temp_audio_filepath)
        output_txt_path = temp_audio_filepath + ".txt"
        if os.path.exists(output_txt_path):
            os.remove(output_txt_path)
