# Databricks notebook source
# DBTITLE 1,Introduction - Voice Conversion
# MAGIC %md
# MAGIC # Voice Conversion: Singer Voice Cloning
# MAGIC
# MAGIC ## 🎤 Goal
# MAGIC
# MAGIC Aap apni voice mein gaate ho, aur wo **singer ki voice mein convert** ho jaye!
# MAGIC
# MAGIC **Example**:
# MAGIC - **Input**: Tumhari voice mein gaana
# MAGIC - **Target**: Arijit Singh / Shreya Ghoshal / koi bhi singer
# MAGIC - **Output**: Tumhara gaana singer ki voice mein
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧠 What is Voice Conversion?
# MAGIC
# MAGIC **Voice Conversion (VC)** ek technique hai jismein:
# MAGIC - **Source speaker** ki voice ko
# MAGIC - **Target speaker** ki voice mein convert karte hain
# MAGIC - **Content** same rahta hai (words/melody)
# MAGIC - **Voice characteristics** change hote hain (timbre, tone, style)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Key Concepts
# MAGIC
# MAGIC ### 1. Timbre (Voice Quality)
# MAGIC - Voice ki unique "color" or "texture"
# MAGIC - Jo aapki voice ko dusre se alag banata hai
# MAGIC - Speaker identity define karta hai
# MAGIC
# MAGIC ### 2. Pitch (Sur)
# MAGIC - How high or low the voice is
# MAGIC - Musical notes (Sa, Re, Ga...)
# MAGIC - Melody define karta hai
# MAGIC
# MAGIC ### 3. Prosody (Speaking/Singing Style)
# MAGIC - Rhythm, stress, intonation
# MAGIC - Emotional expression
# MAGIC - Singing technique
# MAGIC
# MAGIC **Voice Conversion**: Timbre change karo, pitch/prosody preserve karo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Popular Approaches
# MAGIC
# MAGIC ### 1. **RVC (Retrieval-based Voice Conversion)**
# MAGIC - Most popular for singing
# MAGIC - Open-source, easy to use
# MAGIC - Good quality results
# MAGIC - **We'll use this!**
# MAGIC
# MAGIC ### 2. **So-VITS-SVC**
# MAGIC - VITS-based singing voice conversion
# MAGIC - High quality
# MAGIC - More complex
# MAGIC
# MAGIC ### 3. **Diff-SVC**
# MAGIC - Diffusion-based
# MAGIC - Cutting-edge quality
# MAGIC - Slower inference
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 What You'll Learn
# MAGIC
# MAGIC 1. Audio preprocessing (loading, resampling)
# MAGIC 2. Feature extraction (pitch, mel-spectrogram)
# MAGIC 3. Voice conversion model setup (RVC)
# MAGIC 4. Training pipeline (if you have target voice samples)
# MAGIC 5. Inference pipeline (convert your voice)
# MAGIC 6. Practical tips and tricks

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Summary & Resources
# MAGIC %md
# MAGIC ## 📋 Summary & Next Steps
# MAGIC
# MAGIC ### What You Learned
# MAGIC
# MAGIC ✅ **Voice Conversion Fundamentals**
# MAGIC - What is voice conversion (changing timbre, preserving pitch/content)
# MAGIC - Key concepts: Pitch (F0), Timbre, Prosody
# MAGIC - Popular approaches: RVC, So-VITS-SVC, Diff-SVC
# MAGIC
# MAGIC ✅ **Audio Preprocessing**
# MAGIC - Load, resample, normalize, trim audio
# MAGIC - Prepare audio for model input
# MAGIC - Quality control and visualization
# MAGIC
# MAGIC ✅ **Feature Extraction**
# MAGIC - Pitch (F0) extraction with Praat/Parselmouth
# MAGIC - Mel-spectrogram for voice texture
# MAGIC - Visualization and analysis
# MAGIC
# MAGIC ✅ **RVC Pipeline**
# MAGIC - HuBERT content encoding
# MAGIC - Retrieval-based conversion
# MAGIC - Training vs pre-trained models
# MAGIC - Parameter tuning
# MAGIC
# MAGIC ✅ **Best Practices**
# MAGIC - Input audio quality requirements
# MAGIC - Common problems and solutions
# MAGIC - Post-processing tips
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Quick Start Guide
# MAGIC
# MAGIC #### For Beginners (Use Pre-trained Model)
# MAGIC
# MAGIC 1. **Clone RVC**:
# MAGIC    ```bash
# MAGIC    git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# MAGIC    cd Retrieval-based-Voice-Conversion-WebUI
# MAGIC    pip install -r requirements.txt
# MAGIC    ```
# MAGIC
# MAGIC 2. **Download Models**:
# MAGIC    - HuBERT base model
# MAGIC    - Pre-trained RVC model for target singer
# MAGIC
# MAGIC 3. **Convert**:
# MAGIC    ```bash
# MAGIC    python infer-web.py  # Launch Web UI
# MAGIC    # Upload your singing, select model, convert!
# MAGIC    ```
# MAGIC
# MAGIC #### For Advanced (Train Custom Model)
# MAGIC
# MAGIC 1. **Collect Data**: 10-60 minutes of target singer's clean audio
# MAGIC 2. **Preprocess**: Use functions from this notebook
# MAGIC 3. **Train**: Follow RVC training guide
# MAGIC 4. **Infer**: Convert your voice with trained model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 🎯 **Voice Conversion** changes WHO is singing, not WHAT is being sung
# MAGIC
# MAGIC 🎯 **Pitch (F0)** must be preserved for correct melody
# MAGIC
# MAGIC 🎯 **Quality In = Quality Out** - use clean, high-quality audio
# MAGIC
# MAGIC 🎯 **RVC** is currently the best open-source option for singing voice conversion
# MAGIC
# MAGIC 🎯 **Training data quality** matters more than quantity
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Resources & Links
# MAGIC
# MAGIC #### Official Repositories
# MAGIC
# MAGIC 1. **RVC (Retrieval-based Voice Conversion)**
# MAGIC    - GitHub: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# MAGIC    - Most popular, easy to use
# MAGIC    - Active community
# MAGIC
# MAGIC 2. **So-VITS-SVC**
# MAGIC    - GitHub: https://github.com/svc-develop-team/so-vits-svc
# MAGIC    - High quality
# MAGIC    - More technical setup
# MAGIC
# MAGIC 3. **Diff-SVC**
# MAGIC    - GitHub: https://github.com/CNChTu/Diffusion-SVC
# MAGIC    - Cutting-edge quality
# MAGIC    - Slower inference
# MAGIC
# MAGIC #### Pre-trained Models
# MAGIC
# MAGIC - **HuggingFace**: Search for "RVC" or singer names
# MAGIC - **RVC Community Discord**: Many shared models
# MAGIC - **Civitai**: AI voice models
# MAGIC
# MAGIC #### Learning Resources
# MAGIC
# MAGIC - **RVC Documentation**: Official guide in repository
# MAGIC - **YouTube Tutorials**: Search "RVC voice conversion tutorial"
# MAGIC - **Audio ML Basics**: librosa tutorials, Praat manual
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC 🚀 **Try it yourself**:
# MAGIC 1. Record or find a clean acapella of your singing
# MAGIC 2. Run preprocessing pipeline from this notebook
# MAGIC 3. Use RVC Web UI for conversion
# MAGIC 4. Experiment with parameters
# MAGIC 5. Share your results!
# MAGIC
# MAGIC 🎯 **Advanced Topics** (Not covered here):
# MAGIC - Real-time voice conversion
# MAGIC - Multi-speaker models
# MAGIC - Custom vocoder training
# MAGIC - Pitch correction and auto-tune
# MAGIC - Emotional style transfer
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ You're Ready!
# MAGIC
# MAGIC You now understand the complete voice conversion pipeline from audio preprocessing to final output. The tools and code in this notebook give you the foundation to:
# MAGIC
# MAGIC - Preprocess any singing audio
# MAGIC - Extract and visualize features
# MAGIC - Understand how RVC works
# MAGIC - Use pre-trained models or train your own
# MAGIC - Debug common issues
# MAGIC
# MAGIC **Go make some amazing voice conversions! 🎤🎶**

# COMMAND ----------

# DBTITLE 1,Practical Tips & Best Practices
# MAGIC %md
# MAGIC ## 💡 Practical Tips & Best Practices
# MAGIC
# MAGIC ### For Best Results
# MAGIC
# MAGIC #### 1. **Input Audio Quality**
# MAGIC
# MAGIC ✅ **DO**:
# MAGIC - Use **high-quality recordings** (WAV, 44.1kHz or higher)
# MAGIC - Record in a **quiet environment** (no background noise)
# MAGIC - Use a **good microphone** (not phone mic if possible)
# MAGIC - **No background music** (acapella only)
# MAGIC - **Clear singing** (don't mumble)
# MAGIC
# MAGIC ❌ **DON'T**:
# MAGIC - Use low-quality MP3s (<128kbps)
# MAGIC - Include background music or instruments
# MAGIC - Use recordings with echo or reverb
# MAGIC - Convert from videos with compression
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 2. **Target Singer Data** (If Training)
# MAGIC
# MAGIC ✅ **Need**:
# MAGIC - **10-60 minutes** of clean audio
# MAGIC - **Diverse samples** (different songs, emotions)
# MAGIC - **Same quality** as your input will be
# MAGIC - **Studio versions** better than live recordings
# MAGIC
# MAGIC ⚠️ **Common Issues**:
# MAGIC - Too little data (<5 minutes) → Poor quality
# MAGIC - Too much similar data → Overfitting
# MAGIC - Low-quality training data → Low-quality output
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3. **Singing Style Matching**
# MAGIC
# MAGIC **Your singing should be somewhat similar to target**:
# MAGIC - ✅ Similar **pitch range** (male to male, female to female works best)
# MAGIC - ✅ Similar **singing technique** helps
# MAGIC - ✅ Clear **pronunciation**
# MAGIC
# MAGIC **Can still work with mismatches**, but quality may vary:
# MAGIC - ⚠️ Male → Female: Use pitch shift (+7 to +12 semitones)
# MAGIC - ⚠️ Female → Male: Use pitch shift (-7 to -12 semitones)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 4. **Post-Processing**
# MAGIC
# MAGIC After voice conversion, you may want to:
# MAGIC
# MAGIC **A. Mix with Instrumental**
# MAGIC ```python
# MAGIC from pydub import AudioSegment
# MAGIC
# MAGIC # Load converted voice and instrumental
# MAGIC vocals = AudioSegment.from_wav("converted_voice.wav")
# MAGIC instrumental = AudioSegment.from_wav("instrumental.wav")
# MAGIC
# MAGIC # Adjust volume if needed
# MAGIC vocals = vocals - 3  # Reduce by 3dB
# MAGIC
# MAGIC # Overlay
# MAGIC final = instrumental.overlay(vocals)
# MAGIC final.export("final_song.wav", format="wav")
# MAGIC ```
# MAGIC
# MAGIC **B. Add Light Effects**
# MAGIC - Light reverb (singing sounds more natural)
# MAGIC - EQ adjustments
# MAGIC - Compression (even out volume)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Common Problems & Solutions
# MAGIC
# MAGIC #### Problem 1: Output sounds robotic
# MAGIC
# MAGIC **Causes**:
# MAGIC - Low-quality input audio
# MAGIC - Pitch extraction failed
# MAGIC - Model undertrained
# MAGIC
# MAGIC **Solutions**:
# MAGIC - ✅ Clean input audio better
# MAGIC - ✅ Try different F0 extraction method
# MAGIC - ✅ Lower index rate (use less retrieval)
# MAGIC - ✅ Train model longer
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Problem 2: Pitch is wrong / melody changed
# MAGIC
# MAGIC **Causes**:
# MAGIC - F0 extraction failed
# MAGIC - Wrong pitch shift parameter
# MAGIC
# MAGIC **Solutions**:
# MAGIC - ✅ Visualize and check F0 extraction
# MAGIC - ✅ Adjust f0_min and f0_max parameters
# MAGIC - ✅ Try CREPE for more accurate F0
# MAGIC - ✅ Manually correct pitch shift
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Problem 3: Output has artifacts / noise
# MAGIC
# MAGIC **Causes**:
# MAGIC - Background noise in input
# MAGIC - Model quality issues
# MAGIC - Vocoder artifacts
# MAGIC
# MAGIC **Solutions**:
# MAGIC - ✅ Use noise reduction on input first
# MAGIC - ✅ Train model with more/better data
# MAGIC - ✅ Adjust filter radius parameter
# MAGIC - ✅ Post-process with audio editor
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Problem 4: Voice doesn't sound like target
# MAGIC
# MAGIC **Causes**:
# MAGIC - Insufficient training data
# MAGIC - Your voice too different from target
# MAGIC - Low index rate
# MAGIC
# MAGIC **Solutions**:
# MAGIC - ✅ Collect more diverse training data
# MAGIC - ✅ Train longer (more epochs)
# MAGIC - ✅ Increase index rate (0.7-0.9)
# MAGIC - ✅ Try different target singer closer to your voice

# COMMAND ----------

# DBTITLE 1,Complete Example Pipeline
# Complete Voice Conversion Pipeline Example

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def complete_voice_conversion_pipeline(input_audio_path, output_path="converted.wav"):
    """
    Complete pipeline demonstrating all preprocessing and feature extraction steps.
    Note: Actual RVC model inference requires the RVC repository and trained models.
    
    This function shows you the complete preprocessing workflow.
    """
    
    print("🎵 VOICE CONVERSION PIPELINE")
    print("="*70)
    
    # ========================================
    # STEP 1: PREPROCESSING
    # ========================================
    print("\n📑 STEP 1: AUDIO PREPROCESSING")
    print("-"*70)
    
    audio, sr = preprocess_audio(
        input_audio_path,
        target_sr=44100,
        output_path="temp_preprocessed.wav"
    )
    
    # ========================================
    # STEP 2: FEATURE EXTRACTION
    # ========================================
    print("\n📑 STEP 2: FEATURE EXTRACTION")
    print("-"*70)
    
    # Extract pitch (F0)
    print("\n[2.1] Pitch Extraction")
    f0, times = extract_pitch(audio, sr, f0_min=80, f0_max=800)
    
    # Extract mel-spectrogram
    print("\n[2.2] Mel-Spectrogram Extraction")
    mel_spec = extract_mel_spectrogram(audio, sr, n_mels=80, n_fft=2048, hop_length=512)
    
    # ========================================
    # STEP 3: VISUALIZATIONS
    # ========================================
    print("\n📑 STEP 3: VISUALIZATIONS")
    print("-"*70)
    
    print("\n[3.1] Waveform")
    visualize_audio(audio, sr, "Preprocessed Audio")
    
    print("\n[3.2] Pitch Contour")
    visualize_pitch(f0, times, audio, sr, "Pitch Analysis")
    
    print("\n[3.3] Mel-Spectrogram")
    visualize_mel_spectrogram(mel_spec, sr, 512, "Mel-Spectrogram")
    
    # ========================================
    # STEP 4: STATISTICS
    # ========================================
    print("\n📑 STEP 4: AUDIO STATISTICS")
    print("-"*70)
    
    duration = len(audio) / sr
    voiced_f0 = f0[f0 > 0]
    
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Total samples: {len(audio):,}")
    print(f"  Pitch range: {np.min(voiced_f0):.1f} - {np.max(voiced_f0):.1f} Hz")
    print(f"  Mean pitch: {np.mean(voiced_f0):.1f} Hz")
    print(f"  Voiced percentage: {100*len(voiced_f0)/len(f0):.1f}%")
    
    # ========================================
    # STEP 5: NEXT STEPS
    # ========================================
    print("\n📑 STEP 5: NEXT STEPS FOR RVC")
    print("-"*70)
    print("""
    Your audio is now preprocessed and analyzed!
    
    To complete voice conversion with RVC:
    
    1. ✅ Clone RVC repository:
       git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    
    2. ✅ Download pre-trained models:
       - HuBERT base model
       - Target singer's RVC model (or train your own)
    
    3. ✅ Run RVC inference:
       python infer.py --input temp_preprocessed.wav --model target_singer.pth
    
    4. ✅ Mix with instrumental (if needed)
    
    5. ✅ Enjoy your converted singing! 🎶
    """)
    
    print("="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    
    # Return features for further processing
    return {
        'audio': audio,
        'sr': sr,
        'f0': f0,
        'mel_spec': mel_spec,
        'duration': duration
    }

# Example usage:
# results = complete_voice_conversion_pipeline("your_singing.wav")
# display(Audio(results['audio'], rate=results['sr']))  # Listen to preprocessed audio

print("✅ Complete pipeline ready!")
print("\n🚀 To run: complete_voice_conversion_pipeline('path/to/your_singing.wav')")

# COMMAND ----------

# DBTITLE 1,RVC Pipeline Overview
# MAGIC %md
# MAGIC ## 🚀 RVC (Retrieval-based Voice Conversion) Pipeline
# MAGIC
# MAGIC ### What is RVC?
# MAGIC
# MAGIC **RVC** = State-of-the-art voice conversion method that uses:
# MAGIC 1. **HuBERT** - Feature extraction (content encoder)
# MAGIC 2. **Vector Quantization** - Discretize features
# MAGIC 3. **Retrieval Database** - Find similar features from target voice
# MAGIC 4. **Decoder** - Generate target voice audio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### RVC Architecture
# MAGIC
# MAGIC ```
# MAGIC Your Voice Audio
# MAGIC     ↓
# MAGIC [HuBERT Encoder] → Extract content features (what you're saying/singing)
# MAGIC     ↓
# MAGIC [F0 Extractor] → Extract pitch separately (preserve melody)
# MAGIC     ↓
# MAGIC [Feature Retrieval] → Find matching features from target singer database
# MAGIC     ↓
# MAGIC [Decoder + NSF-HiFiGAN] → Synthesize audio with target voice
# MAGIC     ↓
# MAGIC Output: Your singing in target singer's voice!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Components
# MAGIC
# MAGIC #### 1. **Content Encoder (HuBERT)**
# MAGIC - Extracts **what** you're singing (phonemes, words)
# MAGIC - Removes speaker identity
# MAGIC - Pre-trained on speech data
# MAGIC
# MAGIC #### 2. **Pitch Extractor (F0)**
# MAGIC - Extracts **melody** (musical notes)
# MAGIC - Preserved from source (your voice)
# MAGIC - Guides prosody in output
# MAGIC
# MAGIC #### 3. **Speaker Embedding**
# MAGIC - Vector representation of target singer's voice
# MAGIC - Learned from target voice samples
# MAGIC - Guides voice conversion
# MAGIC
# MAGIC #### 4. **Vocoder (NSF-HiFiGAN)**
# MAGIC - Converts features back to audio waveform
# MAGIC - High-quality neural vocoder
# MAGIC - Pitch-controllable
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Two Modes
# MAGIC
# MAGIC #### Mode 1: Use Pre-trained Model
# MAGIC - Download pre-trained RVC model for popular singers
# MAGIC - Directly convert your voice
# MAGIC - **Fastest** - no training needed
# MAGIC - Quality depends on model availability
# MAGIC
# MAGIC #### Mode 2: Train Custom Model
# MAGIC - Collect target singer's voice samples (10-60 minutes)
# MAGIC - Train RVC model on those samples
# MAGIC - **Best quality** for specific singer
# MAGIC - Requires GPU training (few hours)

# COMMAND ----------

# DBTITLE 1,Complete Workflow
# MAGIC %md
# MAGIC ## 📋 Complete Voice Conversion Workflow
# MAGIC
# MAGIC ### Option A: Using Pre-trained RVC Model
# MAGIC
# MAGIC #### Step 1: Setup
# MAGIC ```bash
# MAGIC # Clone RVC repository
# MAGIC git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# MAGIC cd Retrieval-based-Voice-Conversion-WebUI
# MAGIC
# MAGIC # Install dependencies
# MAGIC pip install -r requirements.txt
# MAGIC
# MAGIC # Download base models (HuBERT, etc.)
# MAGIC # Follow RVC repository instructions
# MAGIC ```
# MAGIC
# MAGIC #### Step 2: Get/Download Target Singer Model
# MAGIC - Option A: Download pre-trained model from RVC community
# MAGIC - Option B: Train your own (see next section)
# MAGIC
# MAGIC #### Step 3: Prepare Your Voice
# MAGIC ```python
# MAGIC # Use our preprocessing code above
# MAGIC audio, sr = preprocess_audio(
# MAGIC     "your_singing.wav",
# MAGIC     target_sr=44100,
# MAGIC     output_path="preprocessed_input.wav"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### Step 4: Run Inference
# MAGIC ```bash
# MAGIC # Using RVC Web UI
# MAGIC python infer-web.py
# MAGIC
# MAGIC # Or using Python script:
# MAGIC python infer.py \
# MAGIC     --model_path models/singer_name.pth \
# MAGIC     --input preprocessed_input.wav \
# MAGIC     --output converted_output.wav \
# MAGIC     --pitch_shift 0  # Adjust if needed
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Option B: Training Custom RVC Model
# MAGIC
# MAGIC #### Requirements
# MAGIC - **Target voice samples**: 10-60 minutes of clean audio
# MAGIC - **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
# MAGIC - **Time**: 2-6 hours training
# MAGIC
# MAGIC #### Training Steps
# MAGIC
# MAGIC **1. Collect Target Voice Data**
# MAGIC ```python
# MAGIC # Organize target singer's audio files
# MAGIC target_data/
# MAGIC     ├── song1.wav
# MAGIC     ├── song2.wav
# MAGIC     ├── song3.wav
# MAGIC     ...
# MAGIC
# MAGIC # Preprocess all files
# MAGIC import os
# MAGIC import glob
# MAGIC
# MAGIC for audio_file in glob.glob("target_data/*.wav"):
# MAGIC     output_name = f"processed_{os.path.basename(audio_file)}"
# MAGIC     preprocess_audio(audio_file, output_path=f"processed_data/{output_name}")
# MAGIC ```
# MAGIC
# MAGIC **2. Extract Features**
# MAGIC ```bash
# MAGIC # Extract HuBERT features from target voice
# MAGIC python extract_feature.py \
# MAGIC     --data_dir processed_data \
# MAGIC     --model_path hubert_base.pt
# MAGIC ```
# MAGIC
# MAGIC **3. Train RVC Model**
# MAGIC ```bash
# MAGIC # Train on target singer's voice
# MAGIC python train.py \
# MAGIC     --data_dir processed_data \
# MAGIC     --experiment_name singer_name \
# MAGIC     --batch_size 8 \
# MAGIC     --epochs 100 \
# MAGIC     --save_every 10
# MAGIC ```
# MAGIC
# MAGIC **4. Test Inference**
# MAGIC ```bash
# MAGIC # Convert your voice using trained model
# MAGIC python infer.py \
# MAGIC     --model_path experiments/singer_name/checkpoints/best.pth \
# MAGIC     --input your_singing.wav \
# MAGIC     --output result.wav
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Parameters to Tune
# MAGIC
# MAGIC #### 1. **Pitch Shift (Semitones)**
# MAGIC - If your voice is lower/higher than target
# MAGIC - -12 to +12 semitones typical range
# MAGIC - 0 = no shift (preserve your melody exactly)
# MAGIC
# MAGIC #### 2. **Index Rate**
# MAGIC - How much to use retrieval database (0-1)
# MAGIC - Higher = more like target singer
# MAGIC - Lower = preserves more of your characteristics
# MAGIC - Typical: 0.5-0.8
# MAGIC
# MAGIC #### 3. **Filter Radius**
# MAGIC - Median filtering for F0 (0-10)
# MAGIC - Higher = smoother pitch
# MAGIC - Lower = more pitch variation
# MAGIC - Typical: 3-5
# MAGIC
# MAGIC #### 4. **RMS Mix Rate**
# MAGIC - How to blend volume envelopes (0-1)
# MAGIC - Higher = use original volume more
# MAGIC - Lower = use model's volume
# MAGIC - Typical: 0.25

# COMMAND ----------

# DBTITLE 1,Feature Extraction - Mel-Spectrogram
# MAGIC %md
# MAGIC ## 🌊 Step 3: Mel-Spectrogram - Voice Texture
# MAGIC
# MAGIC ### What is Mel-Spectrogram?
# MAGIC
# MAGIC **Mel-Spectrogram** = Time-frequency representation of audio
# MAGIC
# MAGIC - Shows **how much energy** is in different **frequency bands** over **time**
# MAGIC - Captures **timbre** (voice texture/color)
# MAGIC - **This gets converted** to match target singer's voice!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why Mel-Spectrogram?
# MAGIC
# MAGIC **Human hearing** is not linear:
# MAGIC - We're more sensitive to low frequencies
# MAGIC - Less sensitive to high frequencies
# MAGIC - **Mel scale** = Perceptually-motivated frequency scale
# MAGIC
# MAGIC **Voice conversion models** work on mel-spectrograms:
# MAGIC - Input: Your voice mel-spec
# MAGIC - Output: Target singer's mel-spec
# MAGIC - Pitch contour guides the conversion
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Parameters
# MAGIC
# MAGIC - **n_mels**: Number of mel bands (80-128 typical)
# MAGIC - **n_fft**: FFT window size (2048 typical)
# MAGIC - **hop_length**: Time resolution (512 typical = ~11.6ms at 44.1kHz)

# COMMAND ----------

# DBTITLE 1,Mel-Spectrogram Code
def extract_mel_spectrogram(audio, sr, n_mels=80, n_fft=2048, hop_length=512):
    """
    Extract mel-spectrogram from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel frequency bands
        n_fft: FFT window size (samples)
        hop_length: Hop size between frames (samples)
    
    Returns:
        mel_spec_db: Log-scale mel-spectrogram (dB)
    """
    print("🌊 Extracting mel-spectrogram...")
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0  # Power spectrogram
    )
    
    # Convert to log scale (decibels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    print(f"  ✅ Mel-spec shape: {mel_spec_db.shape}")
    print(f"  Mel bands: {n_mels}")
    print(f"  Time frames: {mel_spec_db.shape[1]}")
    print(f"  Frame rate: {sr/hop_length:.1f} Hz ({1000*hop_length/sr:.1f}ms per frame)")
    
    return mel_spec_db

def visualize_mel_spectrogram(mel_spec, sr, hop_length, title="Mel-Spectrogram"):
    """
    Visualize mel-spectrogram.
    """
    plt.figure(figsize=(14, 5))
    
    librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB', label='Amplitude (dB)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency (Hz)')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

def compare_mel_spectrograms(mel1, mel2, sr, hop_length, title1="Source", title2="Target"):
    """
    Compare two mel-spectrograms side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # First mel-spec
    img1 = librosa.display.specshow(
        mel1, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='viridis', ax=ax1
    )
    ax1.set_title(title1)
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
    
    # Second mel-spec
    img2 = librosa.display.specshow(
        mel2, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='viridis', ax=ax2
    )
    ax2.set_title(title2)
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# Example:
# mel_spec = extract_mel_spectrogram(audio, sr)
# visualize_mel_spectrogram(mel_spec, sr, 512, "Your Voice")

print("✅ Mel-spectrogram extraction ready!")
print("\n💡 Tip: Higher n_mels = more detail, but slower processing")

# COMMAND ----------

# DBTITLE 1,Feature Extraction - Pitch
# MAGIC %md
# MAGIC ## 🎼 Step 2: Feature Extraction - Pitch (F0)
# MAGIC
# MAGIC ### What is Pitch (F0)?
# MAGIC
# MAGIC **Pitch** = Fundamental Frequency = Sur
# MAGIC
# MAGIC - Aapki singing ka **melody**
# MAGIC - Musical notes (Sa, Re, Ga, Ma...)
# MAGIC - **MUST be preserved** from your voice
# MAGIC - Voice conversion changes timbre, NOT pitch!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why Extract Pitch?
# MAGIC
# MAGIC **Problem**: If you just convert voice directly, the melody changes!
# MAGIC - Your "Sa" might become their "Re"
# MAGIC - Song ka melody bigad jayega
# MAGIC
# MAGIC **Solution**: Extract F0 separately, preserve it during conversion
# MAGIC - Your melody + Target singer's voice = Perfect conversion
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Pitch Extraction Methods
# MAGIC
# MAGIC 1. **Praat (Parselmouth)** - Robust, good for singing
# MAGIC 2. **WORLD** - Fast, used in RVC
# MAGIC 3. **CREPE** - Deep learning-based, very accurate
# MAGIC
# MAGIC **We'll use Praat/Parselmouth** - easiest to use

# COMMAND ----------

# DBTITLE 1,Pitch Extraction Code
import parselmouth
from parselmouth.praat import call
import numpy as np

def extract_pitch(audio, sr, f0_min=80, f0_max=800):
    """
    Extract pitch (F0) contour from audio.
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        f0_min: Minimum F0 in Hz (lower for male, ~80-100)
        f0_max: Maximum F0 in Hz (higher for female, ~400-800)
    
    Returns:
        f0_values: Pitch contour (Hz at each time frame)
        times: Time stamps
    """
    print("🎶 Extracting pitch (F0)...")
    
    # Create Praat Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # Extract pitch using autocorrelation
    pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
    
    # Sample F0 at regular intervals (10ms)
    f0_values = []
    times = []
    
    for t in np.arange(0, sound.duration, 0.01):
        f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
        times.append(t)
        
        # Handle unvoiced segments (where there's no pitch)
        if f0 and not np.isnan(f0):
            f0_values.append(f0)
        else:
            f0_values.append(0)  # 0 = unvoiced/silence
    
    f0_values = np.array(f0_values)
    times = np.array(times)
    
    # Statistics
    voiced_f0 = f0_values[f0_values > 0]
    if len(voiced_f0) > 0:
        print(f"  ✅ Extracted {len(f0_values)} frames")
        print(f"  Mean F0: {np.mean(voiced_f0):.1f} Hz")
        print(f"  F0 range: {np.min(voiced_f0):.1f} - {np.max(voiced_f0):.1f} Hz")
        print(f"  Voiced frames: {len(voiced_f0)} ({100*len(voiced_f0)/len(f0_values):.1f}%)")
    
    return f0_values, times

def visualize_pitch(f0, times, audio, sr, title="Pitch Contour"):
    """
    Visualize pitch contour along with waveform.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Waveform
    time_audio = np.arange(len(audio)) / sr
    ax1.plot(time_audio, audio, linewidth=0.5, alpha=0.7, color='blue')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Pitch contour (only voiced segments)
    voiced_mask = f0 > 0
    ax2.plot(times[voiced_mask], f0[voiced_mask], linewidth=2, color='red', marker='o', markersize=2)
    ax2.fill_between(times[voiced_mask], f0[voiced_mask], alpha=0.3, color='red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('F0 (Hz)')
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# Example:
# f0, times = extract_pitch(audio, sr, f0_min=80, f0_max=800)
# visualize_pitch(f0, times, audio, sr, "Your Voice - Pitch Analysis")

print("✅ Pitch extraction functions ready!")
print("\n💡 Tip: Adjust f0_min and f0_max based on singer's gender:")
print("   Male: f0_min=80, f0_max=400")
print("   Female: f0_min=150, f0_max=800")

# COMMAND ----------

# DBTITLE 1,Audio Preprocessing
# MAGIC %md
# MAGIC ## 🎵 Step 1: Audio Preprocessing
# MAGIC
# MAGIC ### Why Preprocessing?
# MAGIC
# MAGIC Voice conversion models need:
# MAGIC - **Consistent sample rate** (usually 16kHz, 22.05kHz, or 44.1kHz)
# MAGIC - **Mono audio** (single channel, not stereo)
# MAGIC - **Clean audio** (noise reduction if needed)
# MAGIC - **Normalized amplitude** (consistent volume levels)
# MAGIC
# MAGIC ### Preprocessing Pipeline
# MAGIC
# MAGIC 1. **Load** audio file
# MAGIC 2. **Resample** to target sample rate
# MAGIC 3. **Convert to mono** (if stereo)
# MAGIC 4. **Normalize** amplitude to [-1, 1]
# MAGIC 5. **Trim silence** (optional, but recommended)
# MAGIC 6. **Save** processed audio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Parameters
# MAGIC
# MAGIC - **Sample Rate**: 44100 Hz for singing (high quality)
# MAGIC - **Bit Depth**: 16-bit or 32-bit float
# MAGIC - **Format**: WAV (lossless) preferred over MP3

# COMMAND ----------

# DBTITLE 1,Preprocessing Code
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def preprocess_audio(audio_path, target_sr=44100, output_path=None):
    """
    Preprocess audio file for voice conversion.
    
    Args:
        audio_path: Path to input audio (WAV, MP3, etc.)
        target_sr: Target sample rate (Hz)
        output_path: Where to save processed audio (optional)
    
    Returns:
        audio: Processed audio array
        sr: Sample rate
    """
    print(f"🎵 Processing: {audio_path}")
    print("="*60)
    
    # Step 1: Load audio
    print("[1/5] Loading audio...")
    audio, sr = librosa.load(audio_path, sr=None)  # Load with original SR
    duration = len(audio) / sr
    print(f"  ✅ Loaded: {sr} Hz, {duration:.2f}s, {len(audio)} samples")
    
    # Step 2: Resample if needed
    if sr != target_sr:
        print(f"[2/5] Resampling {sr} Hz → {target_sr} Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        print(f"  ✅ Resampled to {target_sr} Hz")
    else:
        print(f"[2/5] Sample rate already {target_sr} Hz")
    
    # Step 3: Convert to mono (librosa.load already does this)
    print("[3/5] Audio is mono ✅")
    
    # Step 4: Normalize amplitude
    print("[4/5] Normalizing amplitude...")
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp  # Normalize to [-1, 1]
        print(f"  ✅ Normalized (max amplitude was {max_amp:.3f})")
    
    # Step 5: Trim silence from start and end
    print("[5/5] Trimming silence...")
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    trimmed_duration = len(audio_trimmed) / sr
    print(f"  ✅ Trimmed: {duration:.2f}s → {trimmed_duration:.2f}s")
    
    # Save if requested
    if output_path:
        sf.write(output_path, audio_trimmed, sr)
        print(f"\n💾 Saved to: {output_path}")
    
    print("="*60)
    print("✅ Preprocessing complete!\n")
    
    return audio_trimmed, sr

def visualize_audio(audio, sr, title="Audio Waveform"):
    """
    Visualize audio waveform.
    """
    plt.figure(figsize=(14, 4))
    time = np.arange(len(audio)) / sr
    plt.plot(time, audio, linewidth=0.5, alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# Example usage:
# audio, sr = preprocess_audio("your_singing.wav", output_path="processed_singing.wav")
# visualize_audio(audio, sr, "Your Voice - Preprocessed")
# display(Audio(audio, rate=sr))  # Play in notebook

print("✅ Preprocessing functions ready!")
print("\n💡 Tip: Use high-quality audio (WAV format, no background music) for best results")

# COMMAND ----------

# DBTITLE 1,Setup - Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC ### Core Libraries Needed
# MAGIC
# MAGIC **Audio Processing**:
# MAGIC - `librosa` - Audio analysis and feature extraction
# MAGIC - `soundfile` - Read/write audio files
# MAGIC - `pydub` - Audio manipulation
# MAGIC - `resampy` - High-quality audio resampling
# MAGIC
# MAGIC **Deep Learning**:
# MAGIC - `torch` - PyTorch for neural network models
# MAGIC - `torchaudio` - Audio processing in PyTorch
# MAGIC - `fairseq` - Pre-trained models (HuBERT, etc.)
# MAGIC
# MAGIC **Voice Conversion Specific**:
# MAGIC - `praat-parselmouth` - Pitch extraction (F0)
# MAGIC - `pyworld` - Vocoder features
# MAGIC - `faiss` - Fast vector search (for RVC retrieval)
# MAGIC
# MAGIC **Utilities**:
# MAGIC - `numpy`, `scipy` - Numerical operations
# MAGIC - `matplotlib` - Visualization

# COMMAND ----------

# DBTITLE 1,Install Packages
# Install all required dependencies
# Run this cell first!

%pip install librosa soundfile pydub resampy
%pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
%pip install praat-parselmouth pyworld faiss-cpu
%pip install numpy scipy matplotlib

print("✅ All dependencies installed!")
print("\n📌 Note: For full RVC implementation, you'll need:")
print("   - Clone RVC repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
print("   - Download pre-trained models (HuBERT, etc.)")
print("\nThis notebook shows you the core pipeline and concepts.")