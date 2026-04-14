import streamlit as st
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tempfile
import os
import time

# Page config
st.set_page_config(
    page_title="Noise Reduction System",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.stApp { background-color: #0a0a0a; color: white; }
h1 { text-align: center; }
.loader {
  border: 4px solid #222;
  border-top: 4px solid white;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: auto;
}
@keyframes spin {
  100% { transform: rotate(360deg); }
}
.loading-text { text-align: center; color: gray; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎧 Noise Reduction System")

# Sidebar
st.sidebar.title("⚙ Controls")
strength = st.sidebar.slider("Noise Reduction Strength", 0.1, 1.0, 0.7)
output_format = st.sidebar.selectbox("Output Format", ["mp3", "wav", "aac"])
show_spec = st.sidebar.checkbox("Show Spectrogram", True)

# Upload
file = st.file_uploader("📤 Upload Audio File", type=["wav", "mp3", "aac"])

def calculate_noise(signal):
    return np.mean(np.abs(signal))

if file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        temp_path = tmp.name

    # Convert to WAV
    audio_segment = AudioSegment.from_file(temp_path)
    wav_path = temp_path + ".wav"
    audio_segment.export(wav_path, format="wav")

    # Load audio
    audio, sr = librosa.load(wav_path, sr=None)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔊 Original Audio")
        st.audio(file)

    # File Info
    st.subheader("📁 File Information")
    file_size = len(file.getvalue()) / 1024
    duration = librosa.get_duration(y=audio, sr=sr)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Format", file.type.split("/")[-1].upper())
    c2.metric("Duration", f"{duration:.2f} sec")
    c3.metric("Sample Rate", f"{sr} Hz")
    c4.metric("Size", f"{file_size:.1f} KB")

    # Loading
    loader = st.empty()
    progress = st.progress(0)

    messages = ["Analyzing...", "Reducing noise...", "Enhancing...", "Finalizing..."]

    for i, msg in enumerate(messages):
        loader.markdown(f"""
        <div class="loader"></div>
        <p class="loading-text">{msg}</p>
        """, unsafe_allow_html=True)
        progress.progress((i+1)*25)
        time.sleep(0.3)

    # Process
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=strength)

    loader.empty()
    progress.empty()

    # Save base WAV
    sf.write("clean.wav", reduced_noise, sr)

    # Convert based on selection
    output_file = f"clean.{output_format}"

    sound = AudioSegment.from_wav("clean.wav")
    sound.export(output_file, format=output_format)

    with col2:
        st.subheader("🎵 Clean Audio")
        st.audio(output_file)

    st.success("✅ Noise Reduction Completed")

    # Metrics
    st.subheader("📊 Noise Analysis")

    original_noise = calculate_noise(audio)
    clean_noise = calculate_noise(reduced_noise)

    m1, m2, m3 = st.columns(3)
    m1.metric("Original", f"{original_noise:.4f}")
    m2.metric("Reduced", f"{clean_noise:.4f}")
    m3.metric("Improvement", f"{(original_noise-clean_noise):.4f}")

    # Spectrogram
    if show_spec:
        st.subheader("📊 Spectrogram")

        s1, s2 = st.columns(2)

        with s1:
            D = librosa.amplitude_to_db(abs(librosa.stft(audio)))
            plt.figure(figsize=(3,2))
            plt.imshow(D, aspect='auto', origin='lower')
            plt.axis('off')
            st.pyplot(plt)
            plt.clf()

        with s2:
            D2 = librosa.amplitude_to_db(abs(librosa.stft(reduced_noise)))
            plt.figure(figsize=(3,2))
            plt.imshow(D2, aspect='auto', origin='lower')
            plt.axis('off')
            st.pyplot(plt)
            plt.clf()

    # Download
    with open(output_file, "rb") as f:
        st.download_button(
            f"⬇ Download Clean Audio ({output_format.upper()})",
            f,
            output_file
        )

    # Cleanup
    os.remove(temp_path)
    os.remove(wav_path)
    os.remove("clean.wav")
    os.remove(output_file)

# Footer
st.markdown("""
<hr>
<center style='color:gray;'>© 2026 Nandan R | Noise Reduction System</center>
""", unsafe_allow_html=True)