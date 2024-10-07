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


With Encodec, you can compress audio into a lightweight format to reduce memory usage and subsequently decompress it to restore the original audio. 

This compression process is facilitated by 8 codebooks, each consisting of integer vectors. Think of these codebooks as representations or embeddings of the audio in integer form. It’s important to note that each successive codebook improves the quality of the audio reconstruction from the previous codebooks. As codebooks are integer vectors, they can be learned by transformer models, which are very efficient in this task. This is what Bark was specifically trained to do.

To be more specific, Bark is made of 4 main models:
- **BarkSemanticModel (also referred to as the ‘text’ model):** a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
- **BarkCoarseModel (also referred to as the ‘coarse acoustics’ model):** a causal autoregressive transformer, that takes as input the results of the BarkSemanticModel model. It aims at predicting the first two audio codebooks necessary for EnCodec.
- **BarkFineModel (the ‘fine acoustics’ model)**, this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
- having predicted all the codebook channels from the EncodecModel, Bark uses it to decode the output audio array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the output sound according to specific predefined voice.

Bark is an highly-controllable text-to-speech model, meaning you can use with various settings, as we are going to see.

Before everything, load the model and its processor.

The processor role here is two-sides:
- It is used to tokenize the input text, i.e. to cut it into small pieces that the model can understand.
- It stores speaker embeddings, i.e voice presets that can condition the generation.

```python
from transformers import BarkModel, BarkProcessor

model = BarkModel.from_pretrained("suno/bark-small")
processor = BarkProcessor.from_pretrained("suno/bark-small")
```
Bark is very versatile and can generate audio conditioned by a speaker embeddings library which can be loaded via the processor.

```python
# add a speaker embedding
inputs = processor("This is a test!", voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()
```
It can also generate ready-to-use multilingual speeches, such as French and Chinese. You can find a list of supported languages here. Unlike MMS, discussed below, it is not necessary to specify the language used, but simply adapt the input text to the corresponding language.

```python
# try it in French, let's also add a French speaker embedding
inputs = processor("C'est un test!", voice_preset="v2/fr_speaker_1")

speech_output = model.generate(**inputs).cpu().numpy()
```
The model can also generate non-verbal communications such as laughing, sighing and crying. You just have to modify the input text with corresponding cues such as [clears throat], [laughter], or ....

```python
inputs = processor(
    "[clears throat] This is a test ... and I just took a long pause.",
    voice_preset="v2/fr_speaker_1",
)

speech_output = model.generate(**inputs).cpu().numpy()
```
Bark can even generate music. You can help by adding ♪ musical notes ♪ around your words.

```python
inputs = processor(
    "♪ In the mighty jungle, I'm trying to generate barks.",
)

speech_output = model.generate(**inputs).cpu().numpy()
```

In addition to all these features, Bark supports batch processing, which means you can process several text entries at the same time, at the expense of more intensive computation. On some hardware, such as GPUs, batching enables faster overall generation, which means it can be faster to generate samples all at once than to generate them one by one.

Let’s try generating a few examples:

```python
input_list = [
    "[clears throat] Hello uh ..., my dog is cute [laughter]",
    "Let's try generating speech, with Bark, a text-to-speech model",
    "♪ In the jungle, the mighty jungle, the lion barks tonight ♪",
]

# also add a speaker embedding
inputs = processor(input_list, voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()
```
Let’s listen to the outputs one by one.

First one:

```python
from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(speech_output[0], rate=sampling_rate)
```
Second one:
```python
Audio(speech_output[1], rate=sampling_rate)
```
Third one:
```python
Audio(speech_output[2], rate=sampling_rate)
```

### Massive Multilingual Speech (MMS)

What if you are looking for a pre-trained model in a language other than English? Massive Multilingual Speech (MMS) is another model that covers an array of speech tasks, however, it supports a large number of languages. For instance, it can synthesize speech in over 1,100 languages.

MMS for text-to-speech is based on VITS Kim et al., 2021, which is one of the state-of-the-art TTS approaches.


VITS is a speech generation network that converts text into raw speech waveforms. It works like a conditional variational auto-encoder, estimating audio features from the input text. First, acoustic features, represented as spectrograms, are generated. The waveform is then decoded using transposed convolutional layers adapted from HiFi-GAN. During inference, the text encodings are upsampled and transformed into waveforms using the flow module and HiFi-GAN decoder. Like Bark, there’s no need for a vocoder, as waveforms are generated directly.


Let’s give MMS a go, and see how we can synthesize speech in a language other than English, e.g. German. First, we’ll load the model checkpoint and the tokenizer for the correct language:

```python
from transformers import VitsModel, VitsTokenizer

# use VitsModel and VitsTokenizer since MMS for text-to-speech is based on the VITS model
model = VitsModel.from_pretrained("facebook/mms-tts-deu")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-deu")

#  pick an example text in German
text_example = (
    "Ich bin Schnappi das kleine Krokodil, komm aus Ägypten das liegt direkt am Nil."
)

# To generate a waveform output, preprocess the text with the tokenizer, and pass it to the model
import torch

inputs = tokenizer(text_example, return_tensors="pt")
input_ids = inputs["input_ids"]


with torch.no_grad():
    outputs = model(input_ids)

speech = outputs["waveform"]

# Let’s listen to it:
from IPython.display import Audio

Audio(speech, rate=16000)

```


# Fine-tuning SpeechT5
Now that you are familiar with the text-to-speech task and internal workings of the SpeechT5 model that was pre-trained on English language data, let’s see how we can fine-tune it to another language.

### House-keeping
Make sure that you have a GPU if you want to reproduce this example:
```python
# check GPU 
nvidia-smi
```
some additional dependencies:
```python
pip install transformers datasets soundfile speechbrain accelerate

# log in to your Hugging Face account 
from huggingface_hub import notebook_login
notebook_login()
```

### The dataset
For this example we’ll take the Dutch (nl) language subset of the [VoxPopuli]() dataset.

```python
from datasets import load_dataset, Audio

dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
len(dataset)

# SpeechT5 expects audio data to have a sampling rate of 16 kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

### Preprocessing the data

Let’s begin by  and loading the appropriate processor that contains both tokenizer, and feature extractor that we will need to prepare the data for training:

```python
from transformers import SpeechT5Processor

# defining the model checkpoint to use
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
```

##### Text cleanup for SpeechT5 tokenization
First, for preparing the text, we’ll need the tokenizer part of the processor, so let’s get it:
```python
tokenizer = processor.tokenizer
```
Let’s take a look at an example:
```python
dataset[0]
```
Output:
```python
{'audio_id': '20100210-0900-PLENARY-3-nl_20100210-09:06:43_4',
 'language': 9,
 'audio': {'path': '/root/.cache/huggingface/datasets/downloads/extracted/02ec6a19d5b97c03e1379250378454dbf3fa2972943504a91c7da5045aa26a89/train_part_0/20100210-0900-PLENARY-3-nl_20100210-09:06:43_4.wav',
  'array': array([ 4.27246094e-04,  1.31225586e-03,  1.03759766e-03, ...,
         -9.15527344e-05,  7.62939453e-04, -2.44140625e-04]),
  'sampling_rate': 16000},
 'raw_text': 'Dat kan naar mijn gevoel alleen met een brede meerderheid die wij samen zoeken.',
 'normalized_text': 'dat kan naar mijn gevoel alleen met een brede meerderheid die wij samen zoeken.',
 'gender': 'female',
 'speaker_id': '1122',
 'is_gold_transcript': True,
 'accent': 'None'}
```

What you may notice is that the dataset examples contain raw_text and normalized_text features. When deciding which feature to use as the text input, it will be important to know that the SpeechT5 tokenizer doesn’t have any tokens for numbers. In normalized_text the numbers are written out as text. Thus, it is a better fit, and we should use normalized_text as input text.

Because SpeechT5 was trained on the English language, it may not recognize certain characters in the Dutch dataset. If left as is, these characters will be converted to <unk> tokens. However, in Dutch, certain characters like à are used to stress syllables. In order to preserve the meaning of the text, we can replace this character with a regular a.

To identify unsupported tokens, extract all unique characters in the dataset using the SpeechT5Tokenizer which works with characters as tokens. To do this, we’ll write the extract_all_chars mapping function that concatenates the transcriptions from all examples into one string and converts it to a set of characters. Make sure to set batched=True and batch_size=-1 in dataset.map() so that all transcriptions are available at once for the mapping function.

```python
def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```
Now you have two sets of characters: one with the vocabulary from the dataset and one with the vocabulary from the tokenizer. To identify any unsupported characters in the dataset, you can take the difference between these two sets. The resulting set will contain the characters that are in the dataset but not in the tokenizer.

```python
dataset_vocab - tokenizer_vocab
```
Output:
```python
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```
To handle the unsupported characters identified in the previous step, we can define a function that maps these characters to valid tokens. Note that spaces are already replaced by ▁ in the tokenizer and don’t need to be handled separately.
```python
replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


dataset = dataset.map(cleanup_text)
```

##### Speakers

The VoxPopuli dataset includes speech from multiple speakers, but how many speakers are represented in the dataset? To determine this, we can count the number of unique speakers and the number of examples each speaker contributes to the dataset. With a total of 20,968 examples in the dataset, this information will give us a better understanding of the distribution of speakers and examples in the data.

```python
from collections import defaultdict

speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1
```
By plotting a histogram you can get a sense of how much data there is for each speaker.
```python
import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()
```

![]()

The histogram reveals that approximately one-third of the speakers in the dataset have fewer than 100 examples, while around ten speakers have more than 500 examples. To improve training efficiency and balance the dataset, we can limit the data to speakers with between 100 and 400 examples.

```python
def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

# Let’s check how many speakers remain:
len(set(dataset["speaker_id"]))

# Let’s see how many examples are left:
len(dataset)
```
You are left with just under 10,000 examples from approximately 40 unique speakers, which should be sufficient.

Note that some speakers with few examples may actually have more audio available if the examples are long. However, determining the total amount of audio for each speaker requires scanning through the entire dataset, which is a time-consuming process that involves loading and decoding each audio file. As such, we have chosen to skip this step here.

##### Speaker embeddings
To enable the TTS model to differentiate between multiple speakers, you’ll need to create a speaker embedding for each example. The speaker embedding is an additional input into the model that captures a particular speaker’s voice characteristics. To generate these speaker embeddings, use the pre-trained spkrec-xvect-voxceleb model from SpeechBrain.

Create a function create_speaker_embedding() that takes an input audio waveform and outputs a 512-element vector containing the corresponding speaker embedding.

```python
import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings
```

It’s important to note that the speechbrain/spkrec-xvect-voxceleb model was trained on English speech from the VoxCeleb dataset, whereas the training examples in this guide are in Dutch. While we believe that this model will still generate reasonable speaker embeddings for our Dutch dataset, this assumption may not hold true in all cases.

For optimal results, we would need to train an X-vector model on the target speech first. This will ensure that the model is better able to capture the unique voice characteristics present in the Dutch language. If you’d like to train your own X-vector model, you can use this script as an example.


### Processing the dataset
Finally, let’s process the data into the format the model expects. Create a prepare_dataset function that takes in a single example and uses the SpeechT5Processor object to tokenize the input text and load the target audio into a log-mel spectrogram. It should also add the speaker embeddings as an additional input.


```python
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

# Verify the processing is correct by looking at a single example:
processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())

# Speaker embeddings should be a 512-element vector:
processed_example["speaker_embeddings"].shape
```
Output:


The labels should be a log-mel spectrogram with 80 mel bins.

```python
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()
```
Side note: If you find this spectrogram confusing, it may be due to your familiarity with the convention of placing low frequencies at the bottom and high frequencies at the top of a plot. However, when plotting spectrograms as an image using the matplotlib library, the y-axis is flipped and the spectrograms appear upside down.

Now we need to apply the processing function to the entire dataset. This will take between 5 and 10 minutes.

```python
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

You’ll see a warning saying that some examples in the dataset are longer than the maximum input length the model can handle (600 tokens). Remove those examples from the dataset. Here we go even further and to allow for larger batch sizes we remove anything over 200 tokens.

```python
def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
len(dataset)


create a basic train/test split:
dataset = dataset.train_test_split(test_size=0.1)
```
Output:


### Data collator
In order to combine multiple examples into a batch, you need to define a custom data collator. This collator will pad shorter sequences with padding tokens, ensuring that all examples have the same length. For the spectrogram labels, the padded portions are replaced with the special value -100. This special value instructs the model to ignore that part of the spectrogram when calculating the spectrogram loss.

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)
```

In SpeechT5, the input to the decoder part of the model is reduced by a factor 2. In other words, it throws away every other timestep from the target sequence. The decoder then predicts a sequence that is twice as long. Since the original target sequence length may be odd, the data collator makes sure to round the maximum length of the batch down to be a multiple of 2.


### Train the model
Load the pre-trained model from the same checkpoint as you used for loading the processor:
```python
from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)


# Disable the use_cache=True option for training, and re-enable cache for generation to speed-up inference time:
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)
```
Define the training arguments. Here we are not computing any evaluation metrics during the training process, we’ll talk about evaluation later in this chapter. Instead, we’ll only look at the loss:

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)
```
Instantiate the Trainer object and pass the model, dataset, and data collator to it.
```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)
```
And with that, we’re ready to start training! Training will take several hours. Depending on your GPU, it is possible that you will encounter a CUDA “out-of-memory” error when you start training. In this case, you can reduce the per_device_train_batch_size incrementally by factors of 2 and increase gradient_accumulation_steps by 2x to compensate.

```python
trainer.train()

# Push the final model to the 🤗 Hub:
trainer.push_to_hub()
```

### Inference
Once you have fine-tuned a model, you can use it for inference! Load the model from the 🤗 Hub (make sure to use your account name in the following code snippet):

```python
model = SpeechT5ForTextToSpeech.from_pretrained(
    "YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl"
)

# Pick an example, here we’ll take one from the test dataset. Obtain a speaker embedding.
example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

# Define some input text and tokenize it.
text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"

inputs = processor(text=text, return_tensors="pt")


# Instantiate a vocoder and generate speech:
from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)


# Ready to listen to the result?
from IPython.display import Audio

Audio(speech.numpy(), rate=16000)
```

Obtaining satisfactory results from this model on a new language can be challenging. The quality of the speaker embeddings can be a significant factor. Since SpeechT5 was pre-trained with English x-vectors, it performs best when using English speaker embeddings. If the synthesized speech sounds poor, try using a different speaker embedding.

Increasing the training duration is also likely to enhance the quality of the results. Even so, the speech clearly is Dutch instead of English, and it does capture the voice characteristics of the speaker (compare to the original audio in the example). Another thing to experiment with is the model’s configuration. For example, try using config.reduction_factor = 1 to see if this improves the results.

# Evaluating text-to-speech models

During the training time, text-to-speech models optimize for the mean-square error loss (or mean absolute error) between the predicted spectrogram values and the generated ones. Both MSE and MAE encourage the model to minimize the difference between the predicted and target spectrograms. However, since TTS is a one-to-many mapping problem, i.e. the output spectrogram for a given text can be represented in many different ways, the evaluation of the resulting text-to-speech (TTS) models is much more difficult.

Unlike many other computational tasks that can be objectively measured using quantitative metrics, such as accuracy or precision, evaluating TTS relies heavily on subjective human analysis.

One of the most commonly employed evaluation methods for TTS systems is conducting qualitative assessments using mean opinion scores (MOS). MOS is a subjective scoring system that allows human evaluators to rate the perceived quality of synthesized speech on a scale from 1 to 5. These scores are typically gathered through listening tests, where human participants listen to and rate the synthesized speech samples.

One of the main reasons why objective metrics are challenging to develop for TTS evaluation is the subjective nature of speech perception. Human listeners have diverse preferences and sensitivities to various aspects of speech, including pronunciation, intonation, naturalness, and clarity. Capturing these perceptual nuances with a single numerical value is a daunting task. At the same time, the subjectivity of the human evaluation makes it challenging to compare and benchmark different TTS systems.

Furthermore, this kind of evaluation may overlook certain important aspects of speech synthesis, such as naturalness, expressiveness, and emotional impact. These qualities are difficult to quantify objectively but are highly relevant in applications where the synthesized speech needs to convey human-like qualities and evoke appropriate emotional responses.

In summary, evaluating text-to-speech models is a complex task due to the absence of one truly objective metric. The most common evaluation method, mean opinion scores (MOS), relies on subjective human analysis. While MOS provides valuable insights into the quality of synthesized speech, it also introduces variability and subjectivity.


































































