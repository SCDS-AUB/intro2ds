---
layout: default
title: "Module 8: Machines That Hear - Audio and Signal Processing"
---

# Module 8: Machines That Hear - Audio and Signal Processing

## Introduction

Sound is vibration traveling through air, a continuous wave of pressure changes that our ears translate into the rich tapestry of human experience—music, speech, the warning cry of a child, the whisper of wind through trees. For millennia, sound remained ephemeral: once produced, it vanished forever into silence. Then, in the span of barely 150 years, humanity learned not only to capture sound but to analyze it, manipulate it, and teach machines to understand it.

This module explores the journey from Thomas Edison's first crackling "Mary Had a Little Lamb" on a tinfoil cylinder to modern AI systems that can transcribe speech, identify songs from ambient noise, generate synthetic voices indistinguishable from humans, and even compose original music. At the heart of this revolution lies a mathematical discovery made two centuries ago by a French mathematician studying heat.

---

## Part 1: The Mathematics of Sound - Fourier's Gift

### Jean-Baptiste Joseph Fourier (1768-1830)

The story of audio signal processing begins not with sound but with heat. Joseph Fourier was born in Auxerre, France, the son of a tailor who died when Fourier was nine. Orphaned and poor, he found refuge in mathematics, eventually becoming a professor and later accompanying Napoleon on his Egyptian campaign, where he helped establish the scientific study of Egyptian artifacts.

But Fourier's immortal contribution came from his study of heat conduction. In 1807, he presented a paper to the French Academy claiming something that seemed almost magical: any periodic function, no matter how complex or jagged, could be represented as a sum of simple sine and cosine waves. The Academy was skeptical—Lagrange particularly objected—and it took until 1822 for Fourier to publish his full theory in "Théorie analytique de la chaleur" (The Analytical Theory of Heat).

### The Fourier Transform

Fourier's insight was profound: complexity can be decomposed into simplicity. A complex wave—whether describing heat distribution in a metal bar or the sound of a violin—is actually the sum of many simple waves, each with its own frequency and amplitude.

Consider the sound of a violin playing the note A (440 Hz). It's not a pure 440 Hz tone—that would sound thin and lifeless, like a tuning fork. Instead, it contains the fundamental frequency (440 Hz) plus a series of harmonics (880 Hz, 1320 Hz, 1760 Hz, and so on), each at different amplitudes. This unique "recipe" of harmonics is what gives the violin its characteristic timbre, distinguishing it from a flute or piano playing the same note.

The Fourier Transform is the mathematical operation that takes a signal in the time domain (amplitude changing over time) and reveals its frequency domain representation (which frequencies are present and at what strength). The inverse transform goes the other way, reconstructing the time signal from its frequency components.

### From Analog to Digital: Harry Nyquist and Claude Shannon

For Fourier analysis to be useful for digital audio, we needed to understand how to sample continuous signals. Two engineers at Bell Labs provided the answer:

**Harry Nyquist** (1889-1976), a Swedish-American engineer, showed in the 1920s that to accurately capture a signal, you must sample it at least twice as fast as its highest frequency component. This became known as the Nyquist rate.

**Claude Shannon** (1916-2001), the father of information theory, rigorously proved this sampling theorem in 1949. Human hearing extends to roughly 20,000 Hz, which is why CD audio samples at 44,100 Hz—just over twice the limit of human perception.

---

## Part 2: Capturing and Storing Sound

### Thomas Edison and the Phonograph (1877)

"Mary had a little lamb, its fleece was white as snow..."

These were the first words ever recorded and played back, spoken by Thomas Edison in December 1877 into his newly invented phonograph. The device was brutally simple: a horn channeled sound onto a diaphragm connected to a needle, which etched grooves into tinfoil wrapped around a rotating cylinder. Playing it back reversed the process—the needle following the grooves made the diaphragm vibrate, reproducing the sound.

Edison initially saw the phonograph as a business machine for dictation. He completely missed its potential for music. That insight would come from others, eventually leading to the vinyl record, the tape recorder, and the CD.

### Digital Audio: From PCM to MP3

The conversion from analog sound to digital data uses **Pulse Code Modulation (PCM)**:
1. **Sample** the audio at regular intervals (e.g., 44,100 times per second)
2. **Quantize** each sample to a discrete value (e.g., 16 bits = 65,536 possible values)
3. **Encode** the sequence of values as binary data

Uncompressed digital audio is massive: one minute of CD-quality stereo is about 10 MB. The need for compression led to one of the most successful audio technologies ever:

**MP3 (MPEG-1 Audio Layer III)**: Developed through the 1980s-90s by a team led by Karlheinz Brandenburg at the Fraunhofer Institute in Germany, MP3 exploits psychoacoustic principles—the quirks of human hearing. We can't hear sounds that are masked by louder nearby frequencies. We're less sensitive to certain frequencies. MP3 discards the "inaudible" data, achieving 10:1 compression with minimal perceived quality loss.

---

## Part 3: Making Machines Understand Speech

### Early Dreams of Talking Machines

The dream of machines that understand speech is ancient. In the 18th century, Wolfgang von Kempelen built a mechanical speaking machine that could produce vowels and some consonants using bellows, resonators, and a leather "mouth."

The modern era of speech recognition began at Bell Labs in the 1950s:

**Audrey (1952)**: Built by Davis, Biddulph, and Balashek at Bell Labs, Audrey could recognize spoken digits—but only from a single speaker, in a carefully controlled environment, and it filled an entire room.

### Hidden Markov Models: The Statistical Revolution

The breakthrough in speech recognition came not from better acoustics but from better statistics. In the late 1970s and 1980s, researchers at IBM, led by Frederick Jelinek (1932-2010), applied **Hidden Markov Models (HMMs)** to speech:

Speech is modeled as a sequence of hidden states (phonemes) that produce observable outputs (acoustic features). The "hidden" aspect reflects the fact that we observe the sound, not the underlying phonetic units directly. HMMs provide a principled way to:
- **Train** the model: learn the probability distributions from labeled data
- **Decode**: given new audio, find the most likely sequence of words

Jelinek famously quipped: "Every time I fire a linguist, the performance of the speech recognizer goes up." This reflected the power of statistical approaches over rule-based linguistic analysis.

### The Deep Learning Revolution

Starting around 2010, deep learning transformed speech recognition:

**Deep Neural Networks for Acoustic Modeling (2010-2012)**: Geoffrey Hinton's group at Toronto, working with Microsoft and IBM, showed that deep neural networks dramatically outperformed HMMs for acoustic modeling.

**Recurrent Neural Networks and LSTMs**: Long Short-Term Memory networks, invented by Sepp Hochreiter and Jürgen Schmidhuber in 1997, proved ideal for sequential data like speech.

**End-to-End Models**: Systems like Deep Speech (Baidu, 2014) and Listen, Attend and Spell (Google, 2015) eliminated the traditional pipeline, directly mapping audio to text.

**Transformers and Wav2Vec**: Facebook AI's Wav2Vec 2.0 (2020) and OpenAI's Whisper (2022) represent the current state of the art, achieving human-parity transcription across many languages.

---

## Part 4: Music Information Retrieval - Finding Songs in Sound

### The Shazam Story

In 1999, Avery Wang and Chris Barton faced an impossible problem: how do you identify a song from a 10-second clip captured on a cell phone in a noisy bar? The audio would be distorted, partial, and competing with conversation, clinking glasses, and other background noise.

Their solution, now known as **audio fingerprinting**, was elegant:

1. **Create spectrograms**: Convert audio to time-frequency representations
2. **Find peaks**: Identify the loudest points in the spectrogram—these are robust to noise
3. **Create fingerprints**: Hash pairs of peaks and their time differences into compact codes
4. **Build a database**: Fingerprint millions of songs, storing the fingerprints in a searchable database
5. **Match**: Fingerprint the query audio and find matching sequences in the database

The genius was in what they *didn't* try to understand. They didn't identify melody, rhythm, or key. They simply found a robust way to match acoustic patterns. In 2018, Apple acquired Shazam for $400 million.

### Music Classification and Generation

Beyond identification, audio ML enables:

**Genre Classification**: Using features like tempo, spectral properties, and MFCCs (Mel-Frequency Cepstral Coefficients, which capture the "shape" of sound in a way inspired by human perception)

**Mood Detection**: Training models on labeled data to identify emotional content in music

**Music Generation**: From WaveNet (DeepMind, 2016) generating raw audio to Jukebox (OpenAI, 2020) creating music with lyrics, to Suno and AIVA creating complete compositions.

---

## Part 5: The Spectrogram - Seeing Sound

### Making the Invisible Visible

A **spectrogram** is a visual representation of sound showing time on the x-axis, frequency on the y-axis, and intensity as color or brightness. It transforms the one-dimensional waveform into a two-dimensional image, revealing patterns invisible to the ear.

The development of the sound spectrograph at Bell Labs in the 1940s was driven by wartime needs—analyzing enemy communications, developing better voice transmission. But it became an essential tool for linguistics, bioacoustics (studying animal sounds), and audio engineering.

### Mel Spectrograms and MFCCs

Human perception of pitch is not linear—we perceive the difference between 100 Hz and 200 Hz as the same "distance" as between 1000 Hz and 2000 Hz (both are octaves). The **Mel scale** captures this perceptual nonlinearity.

**Mel-Frequency Cepstral Coefficients (MFCCs)** compress the mel spectrogram into a compact representation that captures the essential acoustic features. Developed in the 1980s, MFCCs became the standard features for speech and audio analysis for decades.

---

## Part 6: Voice Synthesis - Teaching Machines to Speak

### From Voder to WaveNet

At the 1939 World's Fair in New York, Bell Labs demonstrated the **Voder** (Voice Operating Demonstrator), the first electronic speech synthesizer. A trained operator used a keyboard and pedals to control the synthetic voice—it was like playing speech as an instrument.

The path from Voder to natural-sounding synthesis was long:

**Formant Synthesis (1950s-1980s)**: Modeling the resonant frequencies (formants) of the vocal tract. The result sounded robotic but intelligible—think early GPS voices.

**Concatenative Synthesis (1990s-2000s)**: Splicing together recorded fragments of real speech. Better quality but required large databases and produced occasional "glitches" at join points.

**Statistical Parametric Synthesis**: Using HMMs to generate acoustic features, then converting to audio with a vocoder.

**WaveNet (2016)**: DeepMind's autoregressive neural network generated audio sample-by-sample, producing remarkably natural speech. The first version was too slow for real-time use (requiring minutes to generate seconds of audio), but subsequent optimizations made it practical for Google Assistant.

**Neural Vocoders**: Systems like WaveGlow and HiFi-GAN generate high-quality audio efficiently from mel spectrograms.

### Voice Cloning and Deepfakes

Modern systems like Eleven Labs, Resemble AI, and VALL-E can clone voices from just seconds of sample audio. This raises profound ethical questions: How do we verify the authenticity of audio recordings? What happens when anyone's voice can be synthetically reproduced?

---

## Part 7: Audio Data Science Pipeline

### The Modern Audio Processing Stack

1. **Acquisition**: Recording with appropriate sample rate and bit depth
2. **Preprocessing**: Noise reduction, normalization, silence trimming
3. **Feature Extraction**: Spectrograms, MFCCs, embeddings from pretrained models
4. **Modeling**: Classification, regression, generation
5. **Evaluation**: Word Error Rate (WER) for speech, Mean Opinion Score (MOS) for quality

### Libraries and Tools

- **librosa**: The standard Python library for audio analysis
- **PyTorch Audio (torchaudio)**: Audio processing integrated with PyTorch
- **Hugging Face Transformers**: Pre-trained models for speech recognition, speaker identification
- **Whisper**: OpenAI's open-source multilingual speech recognition
- **Essentia**: C++ library with Python bindings for music information retrieval

---

## DEEP DIVE: Shazam and the Invention of Audio Fingerprinting

### The Problem: "What's That Song?"

It's the late 1990s. You're at a party, a coffee shop, or driving in your car when you hear an incredible song—but you have no idea what it is. The DJ doesn't announce it. The barista shrugs. The radio host has moved on. That moment of musical discovery slips away into the void of unknowability.

This experience was universal and frustrating. If you knew some lyrics, you could try searching online (a new possibility in 1999). If you could hum the melody, a musically inclined friend might identify it. But for instrumental music, or when you only caught a brief snippet—you were out of luck.

**Avery Wang** was a PhD student at Stanford studying electrical engineering when he began thinking about this problem. Born in 1974, Wang had grown up fascinated by both music and mathematics. At Stanford, he worked on audio signal processing under Julius Smith, one of the pioneers of the field.

**Chris Barton** was an MBA student at UC Berkeley with a vision for a consumer service that would identify songs. He had the entrepreneurial drive but needed the technical solution.

When they connected, the match was perfect: Barton's business vision plus Wang's signal processing expertise. But they faced a problem that seemed almost impossible to solve.

### The Challenges

Consider what makes audio fingerprinting extraordinarily difficult:

1. **Degradation**: The user is recording audio through a phone microphone in a noisy environment. The signal is distorted, compressed, and mixed with ambient sound.

2. **Partial Matching**: The system receives maybe 10-15 seconds of audio that could be from any point in a 3-5 minute song.

3. **Scale**: There are millions of songs. The database must be comprehensive enough to be useful.

4. **Speed**: Users expect answers in seconds. Searching millions of songs with lossy audio seems computationally intractable.

5. **Robustness**: The system must work regardless of whether the source is a crystal-clear studio recording or a crackling radio in a moving car.

### The Breakthrough Insight

Wang's key insight was counter-intuitive: instead of trying to understand the music (melody, rhythm, harmony), he would focus on finding features that were:

1. **Robust to noise**: Survived even in degraded recordings
2. **Unique enough to discriminate**: Distinguished between different songs
3. **Compact enough to search**: Could be indexed efficiently

He called these features "landmarks" or "anchors"—stable points in the time-frequency representation of audio that could be reliably extracted from both the clean original and the noisy recording.

### The Algorithm: A Technical Deep Dive

Wang published his approach in 2003 in a paper titled "An Industrial-Strength Audio Search Algorithm." Here's how it works:

#### Step 1: Create a Spectrogram

The audio is converted into a spectrogram—a 2D representation with time on the x-axis and frequency on the y-axis. Each point has an intensity representing how much energy is present at that frequency at that moment.

#### Step 2: Find Spectral Peaks

Rather than using the entire spectrogram, Wang extracts only the peaks—points that are louder than their local neighborhood. These peaks are remarkably stable. Even when noise is added or the audio is compressed, the same peaks tend to appear because they represent the dominant energy in the signal.

The peak extraction uses a local maximum filter: a point is a peak if it's larger than all points within some neighborhood (e.g., 20 frequency bins by 20 time frames).

#### Step 3: Create Fingerprints from Peak Pairs

Here's the clever part. A single peak isn't distinctive enough—there might be hundreds of peaks in any audio segment, and different songs could share individual peaks. But *pairs of peaks* are much more distinctive.

For each peak (the "anchor"), Wang looks at nearby peaks (within a "target zone") and creates a hash combining:
- Frequency of the anchor point (f1)
- Frequency of the target point (f2)
- Time difference between them (Δt)

This produces a fingerprint hash like: `hash(f1, f2, Δt)` along with an offset time `t1` (when the anchor appears in the song).

#### Step 4: Build the Database

For each song in the database:
1. Extract the spectrogram
2. Find all peaks
3. Generate all fingerprint hashes
4. Store each hash in a hash table with the song ID and offset time

A 3-minute song might generate 10,000-20,000 fingerprint hashes. With millions of songs, the database contains billions of hashes—but hash table lookups are O(1), so searching is fast.

#### Step 5: Match Query Audio

When a user submits a query:
1. Generate fingerprints from the query audio
2. Look up each fingerprint hash in the database
3. For matches, record the song ID and compute the time offset (database offset - query offset)
4. Songs with many matches at a consistent time offset are candidates

The key insight is the **time coherence** requirement. Random matches will occur—different songs might share some fingerprints. But in the correct song, many fingerprints will match, and they'll all show the same time offset (because the query's position in the song is fixed).

For example, if your 10-second query is from 1:30 to 1:40 in the song, then every matching fingerprint will show an offset of 90 seconds. False matches will have random offsets, filtering them out.

### The Result: Magic in Your Pocket

Shazam launched in the UK in 2002 as a phone service—you'd dial a number, hold your phone up to the music, and receive a text message with the song title. By the time smartphones arrived, Shazam was ready to become an app.

The numbers tell the story:
- By 2014: 10 billion songs identified
- By 2018: Over 1 billion app downloads
- Apple's acquisition price: $400 million

Wang's algorithm proved so robust that it worked in scenarios he never anticipated. Users Shazam'd songs playing from laptop speakers. From TV commercials. From their own humming (a later feature using a different approach). The system identified songs from live concerts, complete with crowd noise.

### Why This Story Matters for Data Science

The Shazam story embodies several crucial data science principles:

1. **Feature Engineering Over Model Complexity**: Wang didn't need neural networks or complex machine learning. The power came from clever feature design—finding the right representation that captured what mattered while discarding what didn't.

2. **Robustness Through Invariance**: By focusing on time-frequency peaks rather than raw audio, the algorithm gained natural invariance to noise and amplitude changes. Good features are robust to irrelevant variations.

3. **Scalability Through Hashing**: The problem seemed to require O(n) search through a database of millions. Hash-based lookup reduced this to O(1) per query fingerprint, making real-time matching possible.

4. **Physical Intuition**: Wang's engineering insight that spectral peaks are stable wasn't derived from machine learning—it came from understanding the physics of audio signals and psychoacoustics.

5. **Less is More**: The algorithm works precisely because it ignores most of the audio signal. By extracting only a few thousand robust features per song, matching becomes tractable.

---

## LECTURE PLAN: From Fourier to Shazam - The Mathematics of Music Recognition

### Learning Objectives
By the end of this lecture, students will be able to:
1. Explain the Fourier Transform and its role in audio analysis
2. Create and interpret spectrograms
3. Understand the concept of audio fingerprinting
4. Implement a simplified audio matching system
5. Appreciate the engineering trade-offs in real-world audio systems

### Lecture Structure (90 minutes)

#### Opening Hook (8 minutes)
**The Mystery Song**
- Play a 10-second audio clip recorded in a noisy environment
- Ask students: "How could a computer identify this song?"
- Demonstrate Shazam identifying it instantly
- Pose the question: "How is this possible? How does it search millions of songs in seconds?"

#### Part 1: The Sound of Data (20 minutes)

**What is Sound? (5 minutes)**
- Sound as pressure waves in air
- Demonstrate with tuning forks or audio software
- The waveform: amplitude vs. time
- Limitations of the time-domain view: "What note is this? What instrument?"

**Fourier's Revolution (10 minutes)**
- Historical context: Fourier studying heat, not sound
- The core idea: any wave = sum of sine waves
- Interactive demo: build complex waves from simple ones
- Live demonstration: show Fourier decomposition of:
  - A pure sine wave
  - A violin note (fundamental + harmonics)
  - Human speech

**From Time to Frequency (5 minutes)**
- The spectrogram: the bridge between time and frequency
- Live demo: visualize different sounds as spectrograms
- Show how different instruments have different "fingerprints"
- Demonstrate the distinctiveness of spectrograms

#### Part 2: Digital Audio Fundamentals (15 minutes)

**Sampling and Quantization (7 minutes)**
- Nyquist theorem: sample at 2x the highest frequency
- CD audio: 44.1 kHz sampling rate, 16-bit depth
- Interactive: what happens with under-sampling (aliasing)
- Compute data rates: why compression matters

**Audio Features (8 minutes)**
- Raw audio vs. extracted features
- Mel scale: how humans perceive pitch
- MFCCs: the "shape" of sound
- Show how features compress information
- Demo: same song in different features

#### Part 3: The Shazam Algorithm (30 minutes)

**The Challenge (5 minutes)**
- The pre-smartphone world: "What's that song?"
- The technical requirements: noise, partial matching, speed
- Why naive approaches fail

**The Solution: Audio Fingerprinting (15 minutes)**
- Step through the algorithm:
  1. Spectrogram generation
  2. Peak detection (draw on board, show examples)
  3. Fingerprint creation (pairs of peaks)
  4. Hash table storage
  5. Matching with time coherence
- Work through example with actual spectrograms
- Show why peak pairs are distinctive
- Demonstrate time coherence filtering

**Scaling Up (10 minutes)**
- Database design: billions of hashes
- Hash table lookup complexity: O(1)
- Distributed systems for real-world deployment
- Handling edge cases: live performances, covers, remixes

#### Part 4: Modern Audio ML (12 minutes)

**Beyond Fingerprinting (4 minutes)**
- Limitations of fingerprinting: must have exact recording
- What about identifying covers? Remixes? Hummed melodies?

**Deep Learning for Audio (8 minutes)**
- Spectrograms as images: CNNs for audio
- Speech recognition: from HMMs to end-to-end
- Music generation: WaveNet, Jukebox, Suno
- Voice synthesis and its ethical implications

#### Wrap-Up and Preview (5 minutes)
- Recap: Fourier → spectrogram → fingerprints → matching
- Key insights: feature engineering, scalability, robustness
- Preview the hands-on exercise
- Open questions: What other applications can you imagine for audio fingerprinting?

### Materials Needed
- Audio playback system with speakers
- Visualization software (Audacity, librosa in Jupyter)
- Pre-prepared spectrograms and audio clips
- Shazam or similar app for demonstration
- Tuning forks (optional, for physical demonstration)

### Discussion Questions
1. Why did Wang use pairs of peaks instead of single peaks for fingerprinting?
2. What would happen if someone recorded a song at a different speed? Would Shazam still work?
3. How might you fingerprint a song to match cover versions?
4. What are the ethical implications of audio fingerprinting technology?

---

## HANDS-ON EXERCISE: Building an Audio Analysis and Mini-Shazam System

### Overview
In this exercise, students will:
1. Load and visualize audio files
2. Generate spectrograms and extract features
3. Implement a simplified audio fingerprinting algorithm
4. Build a small song identification system

### Prerequisites
- Python 3.8+
- Libraries: librosa, numpy, scipy, matplotlib, hashlib
- Audio files: 5-10 short music clips

### Setup

```python
# Install required packages
# pip install librosa numpy scipy matplotlib

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import hashlib

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

### Part 1: Loading and Visualizing Audio (20 minutes)

```python
# Load an audio file
# Note: librosa converts to mono and resamples by default
audio_path = "path/to/your/song.mp3"
y, sr = librosa.load(audio_path, duration=30)  # Load first 30 seconds

print(f"Sample rate: {sr} Hz")
print(f"Audio length: {len(y)} samples")
print(f"Duration: {len(y)/sr:.2f} seconds")

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
```

**Task 1.1**: Load a song and plot its waveform. What do you notice about the amplitude changes over time?

```python
# Create a spectrogram
D = librosa.stft(y)  # Short-Time Fourier Transform
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
```

**Task 1.2**: Generate a spectrogram for your audio. Can you identify the bass line (low frequencies) and higher-pitched elements?

```python
# Create a mel spectrogram (more aligned with human perception)
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

plt.figure(figsize=(14, 6))
librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
```

### Part 2: Feature Extraction (20 minutes)

```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.ylabel('MFCC Coefficient')
plt.tight_layout()
plt.show()

print(f"MFCC shape: {mfccs.shape}")
# Each column is a feature vector representing a short time frame
```

**Task 2.1**: Extract MFCCs from two different songs (or two different genres). How do the patterns differ?

```python
# Extract chroma features (pitch class representation)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

plt.figure(figsize=(14, 5))
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.title('Chroma Features (Pitch Classes)')
plt.tight_layout()
plt.show()
```

### Part 3: Audio Fingerprinting (40 minutes)

Now we'll implement a simplified version of the Shazam algorithm.

```python
def create_spectrogram(y, sr):
    """
    Create a spectrogram for fingerprinting.
    Uses parameters similar to Shazam.
    """
    # Compute spectrogram with specific window size
    hop_length = 512
    n_fft = 2048

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)

    return S, hop_length


def find_peaks(S, neighborhood_size=20):
    """
    Find local peaks in the spectrogram.
    These are points that are louder than their local neighborhood.
    """
    # Apply maximum filter
    local_max = maximum_filter(S, size=neighborhood_size)

    # Peaks are points equal to their local maximum
    # Also apply a threshold to ignore very quiet peaks
    threshold = np.mean(S) + np.std(S)
    peaks = (S == local_max) & (S > threshold)

    # Get coordinates of peaks
    freq_idx, time_idx = np.where(peaks)

    return list(zip(time_idx, freq_idx))


def create_fingerprints(peaks, fan_value=15):
    """
    Create fingerprints from pairs of peaks.
    Each fingerprint is (f1, f2, delta_t) -> (t1)
    """
    fingerprints = []

    # Sort peaks by time
    peaks_sorted = sorted(peaks, key=lambda x: x[0])

    for i, (t1, f1) in enumerate(peaks_sorted):
        # Look at the next 'fan_value' peaks
        for j in range(1, min(fan_value, len(peaks_sorted) - i)):
            t2, f2 = peaks_sorted[i + j]

            # Time difference
            delta_t = t2 - t1

            # Only consider peaks within a reasonable time window
            if delta_t > 0 and delta_t < 200:
                # Create hash from (f1, f2, delta_t)
                fingerprint = (f1, f2, delta_t)
                fingerprints.append((fingerprint, t1))

    return fingerprints


def hash_fingerprint(fingerprint):
    """
    Create a hash from a fingerprint tuple.
    """
    f1, f2, delta_t = fingerprint
    key = f"{f1}|{f2}|{delta_t}"
    return hashlib.md5(key.encode()).hexdigest()
```

Now let's build a database and matching system:

```python
class AudioFingerprinter:
    def __init__(self):
        self.database = defaultdict(list)  # hash -> [(song_id, offset), ...]
        self.song_names = {}  # song_id -> name
        self.song_count = 0

    def add_song(self, audio_path, song_name):
        """Add a song to the database."""
        print(f"Adding: {song_name}")

        # Load audio
        y, sr = librosa.load(audio_path, duration=60)

        # Create spectrogram
        S, hop_length = create_spectrogram(y, sr)

        # Find peaks
        peaks = find_peaks(S)
        print(f"  Found {len(peaks)} peaks")

        # Create fingerprints
        fingerprints = create_fingerprints(peaks)
        print(f"  Created {len(fingerprints)} fingerprints")

        # Store in database
        song_id = self.song_count
        self.song_names[song_id] = song_name
        self.song_count += 1

        for fingerprint, offset in fingerprints:
            h = hash_fingerprint(fingerprint)
            self.database[h].append((song_id, offset))

        return len(fingerprints)

    def identify(self, query_audio):
        """
        Identify a song from a query audio clip.
        Returns the best matching song and confidence score.
        """
        # Load query audio
        if isinstance(query_audio, str):
            y, sr = librosa.load(query_audio, duration=15)
        else:
            y = query_audio
            sr = 22050  # Default sample rate

        # Create spectrogram and find peaks
        S, hop_length = create_spectrogram(y, sr)
        peaks = find_peaks(S)

        # Create fingerprints
        fingerprints = create_fingerprints(peaks)

        # Count matches for each song and offset
        matches = defaultdict(lambda: defaultdict(int))  # song_id -> offset_diff -> count

        for fingerprint, query_offset in fingerprints:
            h = hash_fingerprint(fingerprint)

            if h in self.database:
                for song_id, db_offset in self.database[h]:
                    # The offset difference tells us where in the song the query came from
                    offset_diff = db_offset - query_offset
                    matches[song_id][offset_diff] += 1

        # Find the best match
        best_song = None
        best_count = 0

        for song_id, offset_counts in matches.items():
            max_count = max(offset_counts.values())
            if max_count > best_count:
                best_count = max_count
                best_song = song_id

        if best_song is not None:
            return self.song_names[best_song], best_count
        else:
            return None, 0


# Demo with synthetic audio (since we might not have actual song files)
def create_synthetic_song(freq1, freq2, duration=10, sr=22050):
    """Create a synthetic song with two frequencies and some noise."""
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)
    # Add some noise
    y += 0.1 * np.random.randn(len(y))
    return y, sr
```

**Task 3.1**: Create a database with a few songs and test the matching:

```python
# Create fingerprinter
fp = AudioFingerprinter()

# Add synthetic songs (or replace with real audio paths)
songs = [
    (create_synthetic_song(440, 880), "Song A"),
    (create_synthetic_song(330, 660), "Song B"),
    (create_synthetic_song(523, 1046), "Song C"),
]

for (y, sr), name in songs:
    # Save temporarily and add
    temp_path = f"/tmp/{name.replace(' ', '_')}.wav"
    import soundfile as sf
    sf.write(temp_path, y, sr)
    fp.add_song(temp_path, name)

# Test matching
query_audio, _ = create_synthetic_song(440, 880, duration=3)
result, confidence = fp.identify(query_audio)
print(f"\nIdentified: {result} (confidence: {confidence})")
```

### Part 4: Analysis and Extensions (10 minutes)

**Task 4.1**: Test the robustness of your system by:
- Adding noise to the query
- Using only a portion of the song
- Time-stretching the audio slightly

```python
def add_noise(y, noise_level=0.1):
    """Add Gaussian noise to audio."""
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def test_robustness(fp, original_audio, song_name):
    """Test matching with various degradations."""
    print(f"\nTesting robustness for {song_name}:")

    # Clean audio
    result, conf = fp.identify(original_audio[:int(len(original_audio)*0.3)])
    print(f"  Clean (30% of song): {result} (conf: {conf})")

    # Noisy audio
    noisy = add_noise(original_audio, 0.2)
    result, conf = fp.identify(noisy[:int(len(noisy)*0.3)])
    print(f"  With noise: {result} (conf: {conf})")

    # Very short clip
    result, conf = fp.identify(original_audio[:int(len(original_audio)*0.1)])
    print(f"  10% of song: {result} (conf: {conf})")
```

### Challenge Questions

1. **Feature Selection**: Why do we use spectral peaks instead of other audio features for fingerprinting?

2. **Hash Collisions**: What happens if two different songs produce the same fingerprint hash? How does the algorithm handle this?

3. **Time Invariance**: The algorithm uses pairs of peaks with time differences. Why is this important for matching audio from different positions in a song?

4. **Scaling**: Our implementation stores fingerprints in a Python dictionary. What data structures would you use for a database of millions of songs?

5. **Cover Songs**: This algorithm matches exact recordings. How might you modify it to identify cover versions or different recordings of the same song?

### Expected Outputs

Students should submit:
1. Visualizations of spectrograms for at least 3 different audio sources
2. Analysis of how different audio types produce different spectrograms
3. A working fingerprinting system that can match short clips
4. Performance analysis: accuracy vs. clip length and noise level
5. Written reflection on the strengths and limitations of audio fingerprinting

### Evaluation Rubric

| Criteria | Points |
|----------|--------|
| Correct spectrogram generation and visualization | 20 |
| Working peak detection algorithm | 20 |
| Functional fingerprint creation and matching | 25 |
| Robustness testing and analysis | 20 |
| Code quality and documentation | 15 |
| **Total** | **100** |

---

## Recommended Resources

### Books

**Technical**
- *Digital Signal Processing* by Alan Oppenheim and Ronald Schafer - The classic DSP textbook
- *Speech and Language Processing* by Dan Jurafsky and James Martin - Comprehensive NLP/speech text (free online)
- *Fundamentals of Music Processing* by Meinard Müller - Excellent audio/music analysis book
- *The Scientist and Engineer's Guide to Digital Signal Processing* by Steven Smith - Free online, very accessible

**Historical and Popular**
- *The Information* by James Gleick - Shannon, information theory, and the digital age
- *Chasing Sound: Technology, Culture, and the Art of Studio Recording* by Susan Schmidt Horning
- *How Music Works* by David Byrne - Music, technology, and perception
- *Perfecting Sound Forever* by Greg Milner - The history of recorded music

### Academic Papers

- **Wang, A. (2003)**. "An Industrial-Strength Audio Search Algorithm" - The original Shazam paper
- **Hinton, G., et al. (2012)**. "Deep Neural Networks for Acoustic Modeling in Speech Recognition"
- **Oord, A., et al. (2016)**. "WaveNet: A Generative Model for Raw Audio"
- **Radford, A., et al. (2022)**. "Robust Speech Recognition via Large-Scale Weak Supervision" - Whisper paper
- **Baevski, A., et al. (2020)**. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"

### Video Lectures

- **3Blue1Brown: "But what is the Fourier Transform?"** - Beautiful visual explanation
- **MIT OpenCourseWare: 6.003 Signals and Systems** - Rigorous mathematical treatment
- **Stanford CS224S: Spoken Language Processing** - Dan Jurafsky's course
- **Computerphile: "Audio Fingerprinting"** - Accessible explanation of Shazam

### Online Courses

- **Coursera: Audio Signal Processing for Music Applications** - Stanford course on Coursera
- **Udacity: Digital Signal Processing** - Georgia Tech course
- **Fast.ai: Practical Deep Learning** - Includes audio classification examples

### Tools and Libraries

- **librosa** (https://librosa.org/) - Python audio analysis
- **Essentia** (https://essentia.upf.edu/) - Music information retrieval
- **torchaudio** (https://pytorch.org/audio/) - PyTorch audio
- **Whisper** (https://github.com/openai/whisper) - Open source speech recognition
- **Audacity** (https://www.audacityteam.org/) - Free audio editor with spectrogram view
- **Praat** (https://www.fon.hum.uva.nl/praat/) - Speech analysis software

### Datasets

- **Free Music Archive** (https://freemusicarchive.org/) - Open audio for experimentation
- **LibriSpeech** - Large-scale speech recognition dataset
- **GTZAN Genre Collection** - 1000 audio tracks for genre classification
- **AudioSet** (Google) - Large-scale audio event classification
- **Common Voice** (Mozilla) - Multilingual speech recognition
- **UrbanSound8K** - Urban environmental sounds

---

## References

1. Fourier, J.B.J. (1822). *Théorie analytique de la chaleur*. Paris: Firmin Didot.

2. Shannon, C.E. (1949). "Communication in the Presence of Noise." *Proceedings of the IRE*, 37(1), 10-21.

3. Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm." *Proceedings of the 4th International Conference on Music Information Retrieval*.

4. Davis, K.H., Biddulph, R., & Balashek, S. (1952). "Automatic Recognition of Spoken Digits." *The Journal of the Acoustical Society of America*, 24(6), 637-642.

5. Hinton, G., et al. (2012). "Deep Neural Networks for Acoustic Modeling in Speech Recognition." *IEEE Signal Processing Magazine*, 29(6), 82-97.

6. Oord, A.v.d., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv:1609.03499*.

7. Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." *OpenAI Technical Report*.

8. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." *Proceedings of the 14th Python in Science Conference*.

9. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.

10. Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 257-286.

11. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.

12. Brandenburg, K., & Stoll, G. (1994). "ISO/MPEG-1 Audio: A Generic Standard for Coding of High-Quality Digital Audio." *Journal of the Audio Engineering Society*, 42(10), 780-792.

---

*Module 8 explores how data science enables machines to process and understand the world of sound—from the mathematical foundations of the Fourier Transform to modern AI systems that can transcribe, identify, and generate audio. The story of Shazam demonstrates how elegant algorithms and clever engineering can solve seemingly impossible problems.*
