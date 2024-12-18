import streamlit as st
import tempfile
import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from openai import OpenAI, AuthenticationError
import numpy as np
import re
import ffmpeg

# Remove the pydub import
# from pydub import AudioSegment


def init_openai_client():
    if "openai_client" not in st.session_state:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)

                client.models.list()
                st.session_state.openai_client = client
                st.success("API key is valid!")
            except AuthenticationError:
                st.error("Invalid API key. Please check and try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter your OpenAI API key.")
    return st.session_state.get("openai_client")


@st.cache_data
def transcribe_audio(audio_file_path):
    client = st.session_state.openai_client

    # Convert audio to mp3 if it's not already
    audio = AudioFileClip(audio_file_path)
    mp3_path = tempfile.mktemp(suffix=".mp3")
    audio.write_audiofile(mp3_path)

    with open(mp3_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )

    os.remove(mp3_path)  # Clean up temporary file

    words_with_timestamps = []
    full_transcription = ""

    for segment in transcript.segments:
        full_transcription += segment.text + " "

        # If word-level timestamps are available
        if hasattr(segment, "words") and segment.words:
            for word in segment.words:
                words_with_timestamps.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }
                )
        else:
            # Fallback: Distribute words evenly across the segment duration
            words = segment.text.split()
            segment_duration = segment.end - segment.start
            word_duration = segment_duration / len(words)

            for i, word in enumerate(words):
                start = segment.start + i * word_duration
                end = start + word_duration
                words_with_timestamps.append(
                    {
                        "word": word,
                        "start": start,
                        "end": end,
                    }
                )

    return full_transcription.strip(), words_with_timestamps


@st.cache_data
def correct_text(text, words_with_timestamps):
    client = st.session_state.openai_client

    # First API
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that corrects grammar and removes specific filler words. Maintain the exact structure and pacing of the original text as much as possible.",
            },
            {
                "role": "user",
                "content": f"Please correct the following text by fixing grammatical errors and removing only the words 'um', 'uh', and 'like' when used as filler. Do not add or rearrange words. If no corrections are needed, return the original text unchanged. Return the corrected text, preserving the original word boundaries:\n\n{text}",
            },
        ],
    )
    corrected_text = response.choices[0].message.content.strip()

    # Second API call for timestamp adjustment
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an expert in audio transcription and timestamp correction, specializing in creating perfectly timed transcripts for text-to-speech generation. Your responses should always be in valid JSON format.",
            },
            {
                "role": "user",
                "content": f"""Please adjust the timestamps for the corrected text while maintaining the original pacing. Follow these guidelines:
                    1. Match each word in the corrected text to its corresponding word in the original text.
                    2. Adjust timestamps to ensure natural pacing and prevent word overlaps.
                    3. Allow for brief, natural pauses between sentences and phrases.
                    4. Ensure each word has a duration appropriate for its length and complexity.
                    5. Timestamps should be precise to three decimal places.

                    Original text and timestamps:
                    {text}
                    {words_with_timestamps}

                    Corrected text:
                    {corrected_text}

                    Please provide the output in the following JSON format:
                    {{
                        "text": "corrected text with punctuation",
                        "words": [
                            {{
                                "word": "word",
                                "start": start_time_in_seconds,
                                "end": end_time_in_seconds
                            }},
                            ...
                        ]
                    }}
                """,
            },
        ],
    )

    try:
        response_json = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error(
            "Failed to parse JSON response from OpenAI API. Using original text and timestamps."
        )
        return text, words_with_timestamps

    if (
        not isinstance(response_json, dict)
        or "text" not in response_json
        or "words" not in response_json
    ):
        st.error(
            "Invalid JSON structure in OpenAI API response. Using original text and timestamps."
        )
        return text, words_with_timestamps

    return response_json["text"], response_json["words"]


def generate_speech_with_target_duration(
    client, sentence, target_duration, max_attempts=5
):
    min_speed, max_speed = 0.25, 4.0
    current_speed = 1.0

    for _ in range(max_attempts):
        response = client.audio.speech.create(
            model="tts-1-hd", voice="alloy", input=sentence, speed=current_speed
        )
        temp_audio_path = tempfile.mktemp(suffix=".mp3")
        response.stream_to_file(temp_audio_path)

        audio = AudioFileClip(temp_audio_path)
        current_duration = audio.duration

        if abs(current_duration - target_duration) <= 0.1:
            return audio, temp_audio_path

        speed_ratio = current_duration / target_duration
        current_speed = max(min(current_speed * speed_ratio, max_speed), min_speed)

    return audio, temp_audio_path


def text_to_speech_and_adjust(corrected_text, adjusted_words_with_timestamps):
    client = st.session_state.openai_client
    sentences = re.split(r"(?<=[.!?])\s+", corrected_text)
    sentence_groups = []
    current_sentence = []
    sentence_index = 0

    for word in adjusted_words_with_timestamps:
        current_sentence.append(word)
        if word["word"].strip().endswith((".", "!", "?")):
            sentence_groups.append((sentences[sentence_index], current_sentence))
            current_sentence = []
            sentence_index += 1

    if current_sentence:
        sentence_groups.append((sentences[-1], current_sentence))

    last_word = adjusted_words_with_timestamps[-1]
    final_duration = last_word["end"]
    audio_clips = []

    for sentence, words_in_sentence in sentence_groups:
        start = words_in_sentence[0]["start"]
        end = words_in_sentence[-1]["end"]
        target_duration = end - start

        sentence_audio, temp_audio_path = generate_speech_with_target_duration(
            client, sentence, target_duration
        )

        # Apply fade out
        sentence_audio = sentence_audio.audio_fadeout(0.05)

        # Set the start time for this audio clip
        audio_clips.append(sentence_audio.set_start(start))

        os.remove(temp_audio_path)

    # Combine all audio clips
    final_audio = CompositeAudioClip(audio_clips)
    final_audio = final_audio.set_duration(final_duration)

    return final_audio


def replace_audio(video_path, adjusted_audio):
    video = VideoFileClip(video_path)
    video_with_new_audio = video.set_audio(adjusted_audio)
    output_path = tempfile.mktemp(suffix=".mp4")
    video_with_new_audio.write_videofile(
        output_path, codec="libx264", audio_codec="aac"
    )
    return output_path


def check_ffmpeg():
    try:
        # Try to use ffmpeg-python to run a simple ffmpeg command
        stream = ffmpeg.input("dummy.mp4").output("dummy.mp3")
        ffmpeg.compile(stream)
        return True
    except ffmpeg.Error:
        return False


def extract_audio_from_video(video_file):
    try:
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name

        # Use moviepy to extract audio instead of ffmpeg-python
        video = VideoFileClip(video_file)
        video.audio.write_audiofile(temp_audio_path)

        return temp_audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None


def main():
    st.title("Video Audio Transcription, Correction, and Replacement PoC")

    if not check_ffmpeg():
        st.warning(
            "FFmpeg functionality might be limited. Some features may not work as expected."
        )

    client = init_openai_client()

    if not client:
        return

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        # Extract audio using moviepy
        audio_path = extract_audio_from_video(video_path)

        if audio_path:
            if st.button("Process Video"):
                with st.spinner("Processing..."):
                    transcription, words_with_timestamps = transcribe_audio(audio_path)
                    st.text("Transcription:")
                    st.write(transcription, words_with_timestamps)

                    corrected_text, adjusted_words_with_timestamps = correct_text(
                        transcription, words_with_timestamps
                    )
                    st.text("Corrected Transcription:")
                    st.write(corrected_text, adjusted_words_with_timestamps )

                    adjusted_audio = text_to_speech_and_adjust(
                        corrected_text, adjusted_words_with_timestamps
                    )
                    output_video_path = replace_audio(video_path, adjusted_audio)

                    st.success("Video processing complete!")
                    st.video(output_video_path)

                    st.text("Please try again if quality is bad")

                    # Clean up temporary files
                    os.unlink(video_path)
                    os.unlink(audio_path)
                    os.unlink(output_video_path)
        else:
            st.error("Failed to extract audio from the video.")


if __name__ == "__main__":
    main()
