# I. Text-to-speech datasets

Text-to-speech task (also called speech synthesis) comes with a range of challenges.

First, just like in the previously discussed automatic speech recognition, the alignment between text and speech can be tricky.

However, unlike ASR, TTS is a one-to-many mapping problem, i.e. the same text can be synthesised in many different ways.
Different outputs (spectrograms or audio waveforms) can correspond to the same ground truth. 
The model has to learn to generate the correct duration and timing for each phoneme, word, or sentence which can be challenging, especially for long and complex sentences.

Next, there’s the long-distance dependency problem: language has a temporal aspect, and understanding the meaning of a sentence often requires considering the context of surrounding words. 
Ensuring that the TTS model captures and retains contextual information over long sequences is crucial for generating coherent and natural-sounding speech.

Finally, training TTS models typically requires pairs of text and corresponding speech recordings. 
On top of that, to ensure the model can generate speech that sounds natural for various speakers and speaking styles, data should contain diverse and representative speech samples from multiple speakers.

Collecting such data is expensive, time-consuming and for some languages is not feasible.

You may think, why not just take a dataset designed for ASR and use it for training a TTS model? 
Unfortunately, ASR datasets are not the best option. 
The features that make it beneficial for ASR, such as excessive background noise, are typically undesirable in TTS. 
It’s great to be able to pick out speach from a noisy street recording, but not so much if your voice assistant replies to you with cars honking and construction going full-swing in the background. 
Still, some ASR datasets can sometimes be useful for fine-tuning, as finding top-quality, multilingual, and multi-speaker TTS datasets can be quite challenging.

Let’s explore a few datasets suitable for TTS that you can find on the 🤗 Hub.

### LJSpeech
LJSpeech is a dataset that consists of 13,100 English-language audio clips paired with their corresponding transcriptions. The dataset contains recording of a single speaker reading sentences from 7 non-fiction books in English. LJSpeech is often used as a benchmark for evaluating TTS models due to its high audio quality and diverse linguistic content.

### Multilingual LibriSpeech
Multilingual LibriSpeech is a multilingual extension of the LibriSpeech dataset, which is a large-scale collection of read English-language audiobooks. Multilingual LibriSpeech expands on this by including additional languages, such as German, Dutch, Spanish, French, Italian, Portuguese, and Polish. It offers audio recordings along with aligned transcriptions for each language. The dataset provides a valuable resource for developing multilingual TTS systems and exploring cross-lingual speech synthesis techniques.

### VCTK (Voice Cloning Toolkit)
VCTK is a dataset specifically designed for text-to-speech research and development. It contains audio recordings of 110 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive. VCTK offers a valuable resource for training TTS models with varied voices and accents, enabling more natural and diverse speech synthesis.

### Libri-TTS/ LibriTTS-R
Libri-TTS/ LibriTTS-R is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus. The main differences from the LibriSpeech corpus are listed below:

- The audio files are at 24kHz sampling rate.
- The speech is split at sentence breaks.
- Both original and normalized texts are included.
- Contextual information (e.g., neighbouring sentences) can be extracted.
- Utterances with significant background noise are excluded.

Assembling a good dataset for TTS is no easy task as such dataset would have to possess several key characteristics:

- High-quality and diverse recordings that cover a wide range of speech patterns, accents, languages, and emotions. The recordings should be clear, free from background noise, and exhibit natural speech characteristics.
- Transcriptions: Each audio recording should be accompanied by its corresponding text transcription.
- Variety of linguistic content: The dataset should contain a diverse range of linguistic content, including different types of sentences, phrases, and words. It should cover various topics, genres, and domains to ensure the model’s ability to handle different linguistic contexts.


# Pre-trained models for text-to-speech
Compared to ASR (automatic speech recognition) and audio classification tasks, there are significantly fewer pre-trained model checkpoints available. On the 🤗 Hub, you’ll find close to 300 suitable checkpoints. Among these pre-trained models we’ll focus on two architectures that are readily available for you in the 🤗 Transformers library - SpeechT5 and Massive Multilingual Speech (MMS).

### SpeechT5
SpeechT5 is a model published by Junyi Ao et al. from Microsoft that is capable of handling a range of speech tasks. While in this unit, we focus on the text-to-speech aspect, this model can be tailored to speech-to-text tasks (automatic speech recognition or speaker identification), as well as speech-to-speech (e.g. speech enhancement or converting between different voices). This is due to how the model is designed and pre-trained.

At the heart of SpeechT5 is a regular Transformer encoder-decoder model. Just like any other Transformer, the encoder-decoder network models a sequence-to-sequence transformation using hidden representations. This Transformer backbone is the same for all tasks SpeechT5 supports.

This Transformer is complemented with six modal-specific (speech/text) pre-nets and post-nets. 
- The input speech or text (depending on the task) is preprocessed through a corresponding pre-net to obtain the hidden representations that Transformer can use.
- The Transformer’s output is then passed to a post-net that will use it to generate the output in the target modality.

This is what the architecture looks like:

![]()

SpeechT5 is first pre-trained using large-scale unlabeled speech and text data, to acquire a unified representation of different modalities. During the pre-training phase all pre-nets and post-nets are used simultaneously.

After pre-training, the entire encoder-decoder backbone is fine-tuned for each individual task. At this step, only the pre-nets and post-nets relevant to the specific task are employed. For example, to use SpeechT5 for text-to-speech, you’d need the text encoder pre-net for the text inputs and the speech decoder pre- and post-nets for the speech outputs.

This approach allows to obtain several models fine-tuned for different speech tasks that all benefit from the initial pre-training on unlabeled data.

Note: Even though the fine-tuned models start out using the same set of weights from the shared pre-trained model, the final versions are all quite different in the end. You can’t take a fine-tuned ASR model and swap out the pre-nets and post-net to get a working TTS model, for example. SpeechT5 is flexible, but not that flexible ;)

Let’s see what are the pre- and post-nets that SpeechT5 uses for the TTS task specifically:

- **Text encoder pre-net:** A text embedding layer that maps text tokens to the hidden representations that the encoder expects.
- **Speech decoder pre-net:** This takes a log mel spectrogram as input and uses a sequence of linear layers to compress the spectrogram into hidden representations.
- **Speech decoder post-net:** This predicts a residual to add to the output spectrogram and is used to refine the results.

When combined, this is what SpeechT5 architecture for text-to-speech looks like:

![]()

As you can see, the output is a log mel spectrogram and not a final waveform. It is common for models that generate audio to produce a log mel spectrogram, which needs to be converted to a waveform with an additional neural network known as a vocoder.

Let’s see how you could do that.

First, let’s load the fine-tuned TTS SpeechT5 model from the 🤗 Hub, along with the processor object used for tokenization and feature extraction:

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# tokenize the input text
inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")
```
The SpeechT5 TTS model is not limited to creating speech for a single speaker. Instead, it uses so-called speaker embeddings that capture a particular speaker’s voice characteristics.

**Note:** Speaker embeddings is a method of representing a speaker’s identity in a compact way, as a vector of fixed size, regardless of the length of the utterance. These embeddings capture essential information about a speaker’s voice, accent, intonation, and other unique characteristics that distinguish one speaker from another. Such embeddings can be used for speaker verification, speaker diarization, speaker identification, and more. The most common techniques for generating speaker embeddings include:

- I-Vectors (identity vectors): I-Vectors are based on a Gaussian mixture model (GMM). They represent speakers as low-dimensional fixed-length vectors derived from the statistics of a speaker-specific GMM, and are obtained in unsupervised manner.
- X-Vectors: X-Vectors are derived using deep neural networks (DNNs) and capture frame-level speaker information by incorporating temporal context.
X-Vectors are a state-of-the-art method that shows superior performance on evaluation datasets compared to I-Vectors. The deep neural network is used to obtain X-Vectors: it trains to discriminate between speakers, and maps variable-length utterances to fixed-dimensional embeddings. You can also load an X-Vector speaker embedding that has been computed ahead of time, which will encapsulate the speaking characteristics of a particular speaker.


Let’s load such a speaker embedding from a dataset on the Hub. The embeddings were obtained from the CMU ARCTIC dataset using this script, but any X-Vector embedding should work.



```python
from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch

speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
```
The speaker embedding is a tensor of shape (1, 512). This particular speaker embedding describes a female voice.

At this point we already have enough inputs to generate a log mel spectrogram as an output, you can do it like this:

```python
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

This outputs a tensor of shape (140, 80) containing a log mel spectrogram. The first dimension is the sequence length, and it may vary between runs as the speech decoder pre-net always applies dropout to the input sequence. This adds a bit of random variability to the generated speech.

However, if we are looking to generate speech waveform, we need to specify a vocoder to use for the spectrogram to waveform conversion. In theory, you can use any vocoder that works on 80-bin mel spectrograms. Conveniently, 🤗 Transformers offers a vocoder based on HiFi-GAN. Its weights were kindly provided by the original authors of SpeechT5.

**Note:** HiFi-GAN is a state-of-the-art generative adversarial network (GAN) designed for high-fidelity speech synthesis. It is capable of generating high-quality and realistic audio waveforms from spectrogram inputs.

On a high level, HiFi-GAN consists of one generator and two discriminators. The generator is a fully convolutional neural network that takes a mel-spectrogram as input and learns to produce raw audio waveforms. The discriminators’ role is to distinguish between real and generated audio. The two discriminators focus on different aspects of the audio.

HiFi-GAN is trained on a large dataset of high-quality audio recordings. It uses a so-called adversarial training, where the generator and discriminator networks compete against each other. Initially, the generator produces low-quality audio, and the discriminator can easily differentiate it from real audio. As training progresses, the generator improves its output, aiming to fool the discriminator. The discriminator, in turn, becomes more accurate in distinguishing real and generated audio. This adversarial feedback loop helps both networks improve over time. Ultimately, HiFi-GAN learns to generate high-fidelity audio that closely resembles the characteristics of the training data.


Loading the vocoder is as easy as any other 🤗 Transformers model.
```python
from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```
Now all you need to do is pass it as an argument when generating speech, and the outputs will be automatically converted to the speech waveform.

```python
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
```
Let’s listen to the result. The sample rate used by SpeechT5 is always 16 kHz.

```python
from IPython.display import Audio

Audio(speech, rate=16000)
```

### Bark

Bark is a transformer-based text-to-speech model proposed by Suno AI in suno-ai/bark.

Unlike SpeechT5, Bark generates raw speech waveforms directly, eliminating the need for a separate vocoder during inference – it’s already integrated. This efficiency is achieved through the utilization of Encodec, which serves as both a codec and a compression tool.






































