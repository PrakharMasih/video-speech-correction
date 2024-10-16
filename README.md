# Video Audio Transcription, Correction, and Replacement

This project is a Streamlit application that processes video files by transcribing the audio, correcting the transcription, and replacing the original audio with an improved version. Here's a breakdown of what the code does:

## Main Features

1. **Video Upload**: Users can upload a video file through the Streamlit interface.
2. **Audio Transcription**: The audio from the video is extracted and transcribed using WhisperX.
3. **Text Correction**: The transcription is corrected for grammar and filler words are removed using GPT-4.
4. **Timestamp Adjustment**: The timestamps for the corrected text are adjusted to maintain natural pacing.
5. **Text-to-Speech**: The corrected text is converted back to speech using OpenAI's TTS model.
6. **Audio Replacement**: The original audio in the video is replaced with the new, improved audio.

## How It Works

1. **Transcription** (`transcribe_audio` function):
   - Uses WhisperX to transcribe the audio and generate word-level timestamps.
   - Returns the full transcription and a list of words with their timestamps.

2. **Text Correction** (`correct_text` function):
   - Uses GPT-4 to correct grammar and remove filler words.
   - Adjusts timestamps for the corrected text to maintain natural pacing.

3. **Text-to-Speech** (`generate_speech_with_target_duration` and `text_to_speech_and_adjust` functions):
   - Converts the corrected text back to speech using OpenAI's TTS model.
   - Adjusts the speed of the generated speech to match the original timing.

4. **Audio Processing** (`compress_dynamic_range` function):
   - Applies dynamic range compression to the generated audio for better quality.

5. **Video Processing** (`replace_audio` function):
   - Replaces the original audio in the video with the new, improved audio.

## Main Application Flow

The `main` function sets up the Streamlit interface and orchestrates the entire process:

1. User uploads a video file.
2. When the "Process Video" button is clicked, the application:
   - Extracts audio from the video
   - Transcribes the audio
   - Corrects the transcription
   - Generates new speech from the corrected text
   - Replaces the original audio with the new audio
   - Displays the processed video

## Requirements

- Python libraries: streamlit, moviepy, openai, pydub, whisperx, torch, numpy
- OpenAI API key (for GPT-4 and TTS)
- CUDA-capable GPU (optional, for faster processing)
  
