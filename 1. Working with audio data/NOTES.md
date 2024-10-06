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

**Sampling rate** (sampling frequency): number of samples taken in one second in hertz (Hz).

**Nyquist limit** (half the sampling rate): highest frequency that can be captured from the signal.   
The audible frequencies in human speech are below **8 kHz** and therefore sampling speech at 16 kHz is sufficient.
On the other hand, sampling audio at too low a sampling rate will result in information loss.

**Resampling**: the process of making the sampling rates match.

### 2. Amplitude and bit depth

**Amplitude** (loudness): sound pressure level at any given instant in decibels (dB).

**Bit depth of the sample** determines with how much precision this amplitude value can be described.
In other words, it's a **binary term**, representing the number of possible steps to which the amplitude value can be quantized when it’s converted from continuous to discrete.

The higher the bit depth, the more faithfully the digital representation approximates the original continuous sound wave.

As a consequence, the sampling process introduces noise. The higher the bit depth, the smaller this quantization noise. 
In practice, the quantization noise of 16-bit audio is already small enough to be inaudible, and using higher bit depths is generally not necessary.

Most common audio bit depths:
 - **16-bit and 24-bit**: use integer samples
 - **32-bit**: stores the samples as floating-point values.


Since human hearing is logarithmic in nature — our ears are more sensitive to small fluctuations in quiet sounds than in loud sounds — the loudness of a sound is easier to interpret if the amplitudes are in decibels, which are also logarithmic. 

- **decibel scale for real-world audio**: [**0 dB** (quietest possible sound), **+00** (louder sounds)[.
- **digital audio signals**: ]**-00**, **0 dB** (loudest possible amplitude)].
    
As a quick rule of thumb: every -6 dB is a halving of the amplitude, and anything below -60 dB is generally inaudible unless you really crank up the volume.


### 3. Audio as a waveform

**Waveform** (time domain representation of sound):  plots the sample values over time and illustrates the changes in the sound’s amplitude.

Plotting the waveform for an audio signal with `librosa`:

```python
import librosa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))
```
The example ("trumpet") is loaded as a tuple of audio time series, and sampling rate. 

Let’s take a look at this sound’s waveform by using librosa’s `waveshow()` function:

```python
import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
```

This plots the amplitude of the signal on the y-axis and time along the x-axis. 

In other words, each point corresponds to a single sample value that was taken when this sound was sampled. 
Also note that librosa returns the audio as floating-point values already, and that the amplitude values are indeed within the [-1.0, 1.0] range.


### 4. The frequency spectrum


# II. Load and explore an audio dataset

We will use the **🤗 Datasets* library to work with audio datasets.

Installation:
```python
pip install datasets[audio]
```

The load_dataset() function:
```python
from datasets import load_dataset
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
print(minds)

example = minds[0]
print(example)
```
Output:


the audio column contains several features:
- path: the path to the audio file (*.wav in this case).
- array: The decoded audio data, represented as a 1-dimensional NumPy array.
- sampling_rate. The sampling rate of the audio file (8,000 Hz in this example).

Removing unnecessary features:
```python
columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds
```
Output:


Visualizing the waveform for the first example:
```python
import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
```

Output:



# Preprocessing an audio dataset


