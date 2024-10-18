# Video Audio Transcription, Correction, and Replacement

This project is a Streamlit application that processes video files by transcribing the audio, correcting the transcription, and replacing the original audio with an improved version. Here's a detailed breakdown of the process and code:

## Main Features

1. **Video Upload**: Users can upload a video file through the Streamlit interface.
2. **Audio Transcription**: The audio from the video is extracted and transcribed using OpenAI's Whisper model.
3. **Text Correction**: The transcription is corrected for grammar and filler words are removed using GPT-4.
4. **Timestamp Adjustment**: The timestamps for the corrected text are adjusted to maintain natural pacing.
5. **Text-to-Speech**: The corrected text is converted back to speech using OpenAI's TTS model.
6. **Audio Replacement**: The original audio in the video is replaced with the new, improved audio.

## How It Works

1. **OpenAI Client Initialization** (`init_openai_client` function):
   - Prompts the user to enter their OpenAI API key.
   - Validates the API key and initializes the OpenAI client.

2. **Audio Extraction** (`extract_audio_from_video` function):
   - Uses MoviePy to extract audio from the uploaded video file.

3. **Transcription** (`transcribe_audio` function):
   - Converts the extracted audio to MP3 format.
   - Uses OpenAI's Whisper model to transcribe the audio and generate word-level timestamps.
   - Returns the full transcription and a list of words with their timestamps.

4. **Text Correction** (`correct_text` function):
   - Uses GPT-4 to correct grammar and remove specific filler words ('um', 'uh', 'like').
   - Adjusts timestamps for the corrected text to maintain natural pacing.
   - Returns the corrected text and adjusted word timestamps.

5. **Text-to-Speech** (`generate_speech_with_target_duration` and `text_to_speech_and_adjust` functions):
   - Splits the corrected text into sentences.
   - For each sentence:
     - Generates speech using OpenAI's TTS model.
     - Adjusts the speed of the generated speech to match the original timing.
   - Combines the sentence audio clips into a single audio track.

6. **Audio Replacement** (`replace_audio` function):
   - Uses MoviePy to replace the original audio in the video with the new, improved audio.

7. **Main Application Flow** (`main` function):
   - Sets up the Streamlit interface.
   - Handles file upload and processing button.
   - Orchestrates the entire process from transcription to audio replacement.
   - Displays the processed video.

## Key Components

- **OpenAI API**: Used for transcription (Whisper), text correction (GPT-4), and text-to-speech (TTS-1-HD).
- **MoviePy**: Used for video and audio file manipulation.
- **Streamlit**: Provides the web interface for the application.
- **FFmpeg**: Used indirectly through MoviePy for audio and video processing.

## Error Handling and Edge Cases

- The code includes error handling for API key validation, JSON parsing, and audio extraction.
- It provides fallback mechanisms for word-level timestamps if they're not available from the transcription.
- The text-to-speech function attempts to match the original audio duration through multiple iterations.

## Performance Considerations

- The `@st.cache_data` decorator is used on computationally expensive functions to improve performance through caching.
- Temporary files are used and cleaned up to manage disk space.

## Requirements

- Python libraries: streamlit, moviepy, openai, ffmpeg-python, numpy
- OpenAI API key (for Whisper, GPT-4, and TTS)
- FFmpeg (installed on the system)

## Note

This is a Proof of Concept (PoC) application and may require further optimization for production use. The quality of the output may vary, and users are advised to try again if the initial quality is unsatisfactory.
