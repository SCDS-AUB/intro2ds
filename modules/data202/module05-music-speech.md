---
layout: default
title: "DATA 202 Module 5: Music and Speech Processing"
---

# DATA 202 Module 5: Music and Speech Processing

## Introduction

Audio carries information beyond what's captured in text—emotion in a voice, identity in a song, meaning in intonation. This module explores music information retrieval and speech processing: how machines learn to understand, generate, and transform audio.

Building on Module 8 of DATA 201 (Audio and Signal Processing), we dive deeper into specialized applications: music recommendation, speech recognition, speaker identification, and the emerging world of audio generation.

---

## Part 1: Music Information Retrieval (MIR)

### Understanding Music Computationally

Music is structured sound with:
- **Melody**: Sequence of pitches
- **Harmony**: Simultaneous pitches (chords)
- **Rhythm**: Patterns in time
- **Timbre**: Sound "color" (why a guitar differs from a piano)
- **Structure**: Verses, choruses, bridges

MIR extracts these elements and uses them for:
- **Recommendation**: Suggest similar songs
- **Classification**: Genre, mood, era
- **Transcription**: Convert audio to notation
- **Separation**: Isolate instruments or vocals
- **Generation**: Create new music

### Audio Features for Music

**Low-Level Features**:
- Spectral centroid: Brightness
- Spectral flux: Rate of change
- Zero-crossing rate: Noisiness
- Chroma: Pitch class distribution

**Mid-Level Features**:
- Beats and tempo
- Key and mode
- Chord progressions
- Melodic contour

**High-Level Features**:
- Genre
- Mood/emotion
- Similarity to other songs
- Era/style

### Music Recommendation

**Content-Based**: Analyze audio features, recommend similar
```python
from sklearn.neighbors import NearestNeighbors
# Extract features for all songs
# Find nearest neighbors in feature space
```

**Collaborative Filtering**: Users who liked X also liked Y

**Hybrid**: Combine audio analysis with listening patterns

Spotify combines audio features (extracted via neural networks) with collaborative signals and contextual information (time of day, activity).

---

## Part 2: Speech Recognition

### From Sound to Words

Automatic Speech Recognition (ASR) converts audio to text. The modern pipeline:

1. **Audio Input**: Waveform at 16kHz or higher
2. **Feature Extraction**: Mel spectrograms or learned features
3. **Acoustic Model**: Neural network mapping audio to phonemes or characters
4. **Language Model**: Predict likely word sequences
5. **Decoder**: Combine acoustic and language scores for final transcription

### Modern ASR Architectures

**Encoder-Decoder with Attention**:
- Encode audio with CNN + Transformer
- Decode text autoregressively
- Attention aligns audio to text

**CTC (Connectionist Temporal Classification)**:
- Direct mapping from audio frames to characters
- No explicit alignment needed
- Faster decoding

**Transducer/RNN-T**:
- Combines CTC and attention benefits
- Used in streaming applications

### OpenAI Whisper

**Whisper** (2022) is a breakthrough in ASR:
- Trained on 680,000 hours of web audio
- Multilingual (99 languages)
- Zero-shot: Works without fine-tuning
- Robust to accents, noise, domains

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])

# With language specification
result = model.transcribe("arabic_audio.mp3", language="ar")
```

### Speaker Diarization

"Who spoke when?" involves:
1. Voice activity detection
2. Speaker embedding extraction
3. Clustering similar embeddings
4. Assigning segments to speakers

---

## Part 3: Text-to-Speech and Voice Synthesis

### From Text to Natural Speech

Text-to-Speech (TTS) has evolved dramatically:

**Concatenative (1990s-2000s)**: Stitch recorded speech fragments
**Statistical Parametric (2000s-2010s)**: Generate acoustic features from statistical models
**Neural (2010s-present)**: End-to-end deep learning

### Modern TTS Pipeline

1. **Text Analysis**: Normalize, expand abbreviations
2. **Linguistic Analysis**: Phonemes, prosody prediction
3. **Acoustic Model**: Generate spectrograms (Tacotron, FastSpeech)
4. **Vocoder**: Convert spectrogram to audio (WaveNet, HiFi-GAN)

### Voice Cloning

Modern systems clone voices from minimal samples:
- **VALL-E**: Clone from 3 seconds of audio
- **Eleven Labs**: Commercial voice cloning
- **XTTS**: Open-source multilingual cloning

Ethical concerns:
- Deepfake audio for fraud
- Impersonation
- Consent for voice use

---

## Part 4: Music Generation

### Neural Music Generation

**Symbolic Generation** (MIDI/scores):
- MuseGAN: Generate multi-track music
- Music Transformer: Long-range structure

**Audio Generation**:
- Jukebox (OpenAI): Raw audio with lyrics
- AudioLM (Google): Audio continuation
- MusicGen (Meta): Text-to-music
- Suno: Commercial text-to-music

### Ethical and Legal Issues

- Copyright of training data
- Rights to AI-generated music
- Impact on human musicians
- Authenticity and attribution

---

## DEEP DIVE: The Voice Cloning Revolution

### Three Seconds to Clone a Voice

In January 2023, Microsoft Research unveiled VALL-E: a neural network that could clone any voice from just three seconds of sample audio. The demonstration was striking—the model captured not just the voice but speaking style, accent, and emotional expression.

VALL-E treated speech synthesis as a language modeling problem:
1. Train on 60,000 hours of English speech
2. Learn to predict audio tokens (discrete representations of audio)
3. Condition on a short reference sample
4. Generate arbitrary new speech in that voice

The implications rippled immediately:
- **Positive**: Accessibility tools, voice preservation for those losing their voice
- **Negative**: Voice fraud, fake emergency calls, audio deepfakes

Banks have reported voice-cloning fraud cases. Scammers clone relatives' voices from social media videos to make fake emergency calls.

### The Response

Technology companies face a dilemma:
- Powerful tools enable both beneficial and harmful uses
- Restricting release doesn't prevent bad actors
- Detection tools lag behind generation

Watermarking, detection models, and authentication systems are emerging responses, but the cat-and-mouse game continues.

---

## HANDS-ON EXERCISE: Speech and Music Analysis

### Part 1: Speech Recognition with Whisper

```python
import whisper
import librosa

# Load model (options: tiny, base, small, medium, large)
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("speech.wav")
print(result["text"])

# Word-level timestamps
result = model.transcribe("speech.wav", word_timestamps=True)
for segment in result["segments"]:
    for word in segment.get("words", []):
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

### Part 2: Music Feature Extraction

```python
import librosa
import numpy as np

# Load audio
y, sr = librosa.load("song.mp3")

# Extract features
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

print(f"Tempo: {tempo:.1f} BPM")
print(f"Chroma shape: {chroma.shape}")
print(f"MFCC shape: {mfcc.shape}")
```

### Part 3: Music Similarity

```python
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)
    features = {
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
        'mfcc_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1),
        'chroma_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    }
    return np.concatenate([
        [features['tempo'], features['spectral_centroid']],
        features['mfcc_mean'],
        features['chroma_mean']
    ])

# Extract features for multiple songs
# Find similar songs using nearest neighbors
```

---

## Recommended Resources

### Libraries
- **librosa**: Audio analysis
- **Whisper**: Speech recognition
- **pyannote.audio**: Speaker diarization
- **Coqui TTS**: Text-to-speech

### Courses and Tutorials
- [Music Information Retrieval Course](https://musicinformationretrieval.com/)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course)

### Papers
- "Attention Is All You Need" - Transformers foundation
- "Whisper" - OpenAI's multilingual ASR
- "VALL-E" - Zero-shot TTS
- "MusicGen" - Text-to-music generation

---

*Module 5 explores music and speech processing—the technologies for understanding, analyzing, and generating audio content. From speech recognition to music recommendation to voice synthesis, we learn how machines interact with the auditory world.*
