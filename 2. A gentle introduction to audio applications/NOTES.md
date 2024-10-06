# I. Audio classification with a pipeline

Audio classification involves assigning one or more labels to an audio recording based on its content.
The labels could correspond to different sound categories, such as music, speech, or noise, or more specific categories like bird song or car engine sounds.

Example: the MINDS-14 dataset that contains recordings of people asking an e-banking system questions in several languages and dialects, and has the intent_class for each recording.

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```
To classify an audio recording into a set of classes, we can use the audio-classification pipeline from 🤗 Transformers.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
```
This pipeline expects the audio data as a NumPy array. All the preprocessing of the raw audio data will be conveniently handled for us by the pipeline.

```python
# pick an example 
example = minds[0]

# pass the NumPy array straight to the classifier
classifier(example["audio"]["array"])
```
Output:

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```
Output:


Hooray! The predicted label was correct! 
Here we were lucky to find a model that can classify the exact labels that we need. 
A lot of the times, when dealing with a classification task, a pre-trained model’s set of classes is not exactly the same as the classes you need the model to distinguish. 
In this case, you can fine-tune a pre-trained model to “calibrate” it to your exact set of class labels. We’ll learn how to do this in the upcoming units. 
Now, let’s take a look at another very common task in speech processing, automatic speech recognition.


# II. Automatic speech recognition with a pipeline

Automatic Speech Recognition (ASR) is a task that involves transcribing speech audio recording into text.

In this section, we’ll use the automatic-speech-recognition pipeline to transcribe an audio recording of a person asking a question about paying a bill using the same MINDS-14 dataset as before.

```python
from transformers import pipeline

# instantiate the pipeline
asr = pipeline("automatic-speech-recognition")

#  take an example from the dataset and pass its raw data to the pipeline
example = minds[0]
asr(example["audio"]["array"])
```
Output:


The model seems to have done a pretty good job at transcribing the audio! It only got one word wrong (“card”) compared to the original transcription

By default, this pipeline uses a model trained for automatic speech recognition for English language, which is fine in this example.

If you’d like to try transcribing other subsets of MINDS-14 in different language, you can find a pre-trained ASR model and pass it’s name as the model argument to the pipeline.

When working on solving your own task, starting with a simple pipeline like the ones we’ve shown in this unit is a valuable tool that offers several benefits:

- a pre-trained model may exist that already solves your task really well, saving you plenty of time
- pipeline() takes care of all the pre/post-processing for you
- if the result isn’t ideal, this still gives you a quick baseline for future fine-tuning
- once you fine-tune a model on your custom data and share it on Hub, the whole community will be able to use it quickly and effortlessly via the pipeline() method making AI more accessible.


# III. Audio generation with a pipeline

Audio generation encompasses a versatile set of tasks that involve producing an audio output. 
The tasks that we will look into here are speech generation (aka “text-to-speech”) and music generation. 

- In text-to-speech, a model transforms a piece of text into lifelike spoken language sound, opening the door to applications such as virtual assistants, accessibility tools for the visually impaired, and personalized audiobooks.
- On the other hand, music generation can enable creative expression, and finds its use mostly in entertainment and game development industries.


In 🤗 Transformers, you’ll find a pipeline that covers both of these tasks. 
This pipeline is called "text-to-audio", but for convenience, it also has a "text-to-speech" alias. 


### 1. Generating speech

We’ll define a text-to-speech pipeline since it best describes our task, and use the suno/bark-small checkpoint:
```python
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small")

# passing some text through the pipeline; all the preprocessing will be done for us under the hood:
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)

# In a notebook, we can use the following code snippet to listen to the result:
from IPython.display import Audio
Audio(output["audio"], rate=output["sampling_rate"])
```

The model that we’re using with the pipeline, Bark, is actually multilingual, so we can easily substitute the initial text with a text in, say, French, and use the pipeline in the exact same way. 
It pick up on the language all by itself.

Not only is this model multilingual, it can also generate audio with non-verbal communications and singing. 
Here’s how you can make it sing:
```python
song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
output = pipe(song)
Audio(output["audio"], rate=output["sampling_rate"])
```

# 2. Generating music

For music generation, we’ll define a text-to-audio pipeline, and initialise it with the pretrained checkpoint facebook/musicgen-small

```python
music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")

# create a text description of the music we’d like to generate
text = "90s rock song with electric guitar and heavy drums"

# control the length of the generated output by passing an additional max_new_tokens parameter to the model
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])
```

 




