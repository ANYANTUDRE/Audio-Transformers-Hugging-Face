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













