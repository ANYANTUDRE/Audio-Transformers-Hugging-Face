# I. Introduction to audio data

By nature, **a sound wave is a continuous signal.**
- **Problem:** digital devices expect finite arrays.
- **Solution:** convert signal into a series of discrete values (digital representation).
- **How?**   
           1. The analog signal is first captured by a microphone, which converts the sound waves into an electrical signal.  
           2. The electrical signal is then digitized by an Analog-to-Digital Converter to get the digital representation through **sampling**.
       
The different audio file formats (.wav, .flac, .mp3) mainly differ in how they **compress the digital representation** of the audio signal.

### 1. Sampling and sampling rate

**Sampling:** process of measuring the value of a continuous signal at fixed time steps.
The sampled waveform is **discrete**.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/Signal_Sampling.png)

**Sampling rate** (sampling frequency): number of samples taken in one second in hertz (Hz).

**Nyquist limit** (half the sampling rate): highest frequency that can be captured from the signal.   
The audible frequencies in human speech are below **8 kHz** and therefore sampling speech at 16 kHz is sufficient.
On the other hand, sampling audio at too low a sampling rate will result in information loss.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/nyquist.png)

**Resampling**: the process of making the sampling rates match.

### 2. Amplitude and bit depth

**Amplitude** (loudness): sound pressure level at any given instant in decibels (dB) (ex: a normal speaking voice is under 60 dB)

**Bit depth of the sample** determines with how much precision this amplitude value can be described.
In other words, it's a **binary term**, representing the number of possible steps to which the amplitude value can be quantized when it’s converted from continuous to discrete.

The higher the bit depth, the more faithfully the digital representation approximates the original continuous sound wave.

As a consequence, the sampling process introduces noise. The higher the bit depth, the smaller this quantization noise. 
In practice, the quantization noise of 16-bit audio is already small enough to be inaudible, and using higher bit depths is generally not necessary.

Most common audio bit depths:
 - **16-bit and 24-bit**: use integer samples
 - **32-bit**: stores the samples as floating-point values. Floating-point audio samples are expected to lie within the [-1.0, 1.0] range. Since ML models naturally work on floating-point data, the audio must first be converted into floating-point format before it can be used to train the model.


Since human hearing is logarithmic in nature — our ears are more sensitive to small fluctuations in quiet sounds than in loud sounds — the loudness of a sound is easier to interpret if the amplitudes are **in decibels**, which are also logarithmic. 

- **decibel scale for real-world audio**: [**0 dB** (quietest possible sound), **+00** (louder sounds)[.
- **digital audio signals**: ]**-00**, **0 dB** (loudest possible amplitude)].
    
As a quick rule of thumb: **every -6 dB is a halving of the amplitude**, and anything below -60 dB is generally inaudible unless you really crank up the volume.

### 3. Audio as a waveform

**Waveform** (time domain representation of sound):  plots the sample values over time and illustrates the changes in the sound’s amplitude.

The plot shows the amplitude of the signal on the y-axis and time along the x-axis.  In other words, each point corresponds to a single sample value that was taken when this sound was sampled. Also note that librosa returns the audio as floating-point values already, and that the amplitude values are indeed within the [-1.0, 1.0] range.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/waveform_unit1.png)


### 4. **Frequency Spectrum (Frequency Domain Representation)**
   - **Purpose**: Visualizes the individual frequencies in an audio signal and their strengths.
   - **Method**: Uses the Discrete Fourier Transform (DFT) to calculate frequency components. For efficiency, Fast Fourier Transform (FFT) is commonly applied.
   - **Example**: Calculating DFT with `numpy`’s `rfft()` over the first 4096 samples, typically using a window function to reduce edge effects.
   - **Interpretation**: Peaks in the spectrum show harmonics of a note, with quieter higher harmonics; amplitude in dB and frequency (Hz) are often shown on a logarithmic scale.
             - **Amplitude Spectrum**: Obtained from the magnitude of DFT results; provides strength of frequencies.
             - **Phase Spectrum**: Angle of real and imaginary components; often unused in ML.
             - **Power Spectrum**: Amplitude squared, reflecting energy instead.
   - **Key Note**: Frequency spectrum provides a "snapshot" of frequencies at a specific time, useful for fixed-frequency analysis.
           - **FFT vs. DFT**: The Fast Fourier Transform (FFT) is an efficient way to compute the DFT, and the terms are often used interchangeably.
           - **Waveform vs. Spectrum**: Both representations contain the same information; waveforms show amplitude over time, while spectrums show frequency strengths at a fixed moment.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/spectrum_plot.png)


### 5. **Spectrogram**
   - **Purpose**: Shows how frequencies change over time, offering a fuller picture than the frequency spectrum alone.
   - **Method**: Created by applying Short-Time Fourier Transform (STFT), which divides the audio into small, overlapping segments, each transformed into a frequency spectrum.
   - **Interpretation**: X-axis represents time, Y-axis shows frequency, and color intensity indicates amplitude in dB.
   - **Application**: Useful for identifying instrument sounds, speech patterns, and audio structures over time.
   - **Inversion**: Spectrograms can be converted back to waveforms if phase data is available. Without phase data, algorithms like **Griffin-Lim or vocoder models** reconstruct the waveform.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/spectrogram_plot.png)


### 6. **Mel Spectrogram**
   - **Purpose**: Similar to a spectrogram but uses a **mel scale** to align with human auditory perception, focusing more on lower frequencies as sensitivity decreases logarithmically at higher frequencies.
   - **Method**: The STFT is applied, followed by a mel filterbank that compresses higher frequencies into perceptual bands (converts frequencies to mel scale using filters that mimic human hearing). Often expressed in decibels (log-mel spectrogram).
   - **Example**: Created using `librosa.melspectrogram()`, with parameters like `n_mels` for mel bands and `fmax` for the frequency ceiling.
   - **Variability in Mel Scales**: Common mel scales include "htk" and "slaney," with variations in using power or amplitude spectrograms.
   - **Use Case**: Common in speech recognition, speaker ID, and music analysis because it emphasizes perceptually meaningful features.
   - **Limitations**: Mel spectrograms discard some frequency information, making it hard to reconstruct the waveform directly. Vocoder models like HiFi-GAN help convert mel spectrograms back to audio.

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/mel-spectrogram.png)


##### Key Differences Spectrum vs Spectrogram vs Mel Spectrogram
   - **Frequency Spectrum**: Snapshot of frequencies at a fixed point.
   - **Spectrogram**: Frequencies over time, detailed with amplitude.
   - **Mel Spectrogram**: Frequency information adapted for human perception, important for machine learning tasks. 


# II. Load and explore an audio dataset

We will use the **🤗 Datasets** library to work with audio datasets.

- The load_dataset() function:
```python
from datasets import load_dataset
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
print(minds)

example = minds[0]
print(example)
```

- the audio column contains several features:
           - **path:** the path to the audio file (*.wav in this case).
           - **array:** the decoded audio data, represented as a 1-dimensional NumPy array.
           - **sampling_rate:** the sampling rate of the audio file (8,000 Hz in this example).

# III. Preprocessing an audio dataset

Some general preprocessing steps:
- Resampling the audio data
- Filtering the dataset
- Converting audio data to model’s expected input

### 1. Resampling the audio data
If there’s a discrepancy between the sampling rates of your audios and the model (train or inference), you can resample the audio to the model’s expected sampling rate.
Most of the available pretrained models have been pretrained on audio datasets at a **sampling rate of 16 kHz.**
**Syntax:** `minds = minds.cast_column("audio", Audio(sampling_rate=16_000))`

### 2. Filtering the dataset
You may need to filter the data based on some criteria. One of the common cases involves limiting the audio examples to a certain duration. For instance, we might want to **filter out any examples longer than 20s to prevent out-of-memory errors when training a model.**
**Syntax:** `minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])`

### 3. Pre-processing audio data
The raw audio data comes as an array of sample values but pre-trained models expect it to be **converted into input features** depending on the model.
Transformers offer a `feature extractor` class that can convert raw audio data into the input features the model expects.

**Example of Whisper’s feature extractor operations:**
           - First **padding/truncating a batch of audio examples** such that all examples have an input length of **30s**.  There is no need for an attention mask.
           - Second **converting the padded audio arrays to log-mel spectrograms**. 

The model’s feature extractor class takes care of transforming raw audio data to the format that the model expects. However, many tasks involving audio are multimodal, e.g. speech recognition. In such cases 🤗 Transformers also offer **model-specific tokenizers to process the text inputs.**

You can load the feature extractor and tokenizer for Whisper and other multimodal models separately, or you can load both via a so-called **processor**. To make things even simpler, use **AutoProcessor to load a model’s feature extractor and processor from a checkpoint**, like this:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")
```


# IV. Streaming audio data

One of the biggest challenges faced with audio datasets is their **sheer size**.
- So what happens when we want to train on a larger split?
- Do we need to fork out and buy additional storage? 
- Or is there a way we can train on these datasets with no disk space constraints?

**🤗 Datasets** comes to the rescue by offering the **streaming mode**. Streaming allows us to load the data progressively as we iterate over the dataset. Rather than downloading the whole dataset at once, we load the dataset one example at a time. We iterate over the dataset, loading and preparing examples on the fly when they are needed. This way, we only ever load the examples that we’re using, and not the ones that we’re not! Once we’re done with an example sample, we continue iterating over the dataset and load the next one.

Streaming mode has three primary advantages over downloading the entire dataset at once:
- Disk space
- Download and processing time
- Easy experimentation

There is one caveat to streaming mode. 
- When downloading a full dataset without streaming, both the raw data and processed data are saved locally to disk and this allows reusability.
- With streaming mode, the data is not downloaded to disk. Thus, neither the downloaded nor pre-processed data are cached.

```python
# How can you enable streaming mode? Easy!
gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)

# You can no longer access individual samples using Python indexing. Instead, you have to iterate over the dataset
next(iter(gigaspeech["train"]))

# preview several examples from a large dataset, use the take() to get the first n elements
gigaspeech_head = gigaspeech["train"].take(2)
list(gigaspeech_head)
```
