# I. Pre-trained models and datasets for audio classification

In audio classification, we want to transform a sequence of audio inputs (i.e. our input audio array) into a single class label prediction.
- Hence, there is a preference for encoder-only models for audio classification.
- Decoder-only and Encoder-decoder models introduce unnecessary complexity to the task.

### 🤗 Transformers Installation
To get the the latest updates on the main version of the 🤗 Transformers repository, use the following command:

```python
pip install git+https://github.com/huggingface/transformers
```python

### Keyword Spotting
Keyword spotting (KWS) is the task of identifying a keyword in a spoken utterance. 
The set of possible keywords forms the set of predicted class labels.

##### Minds-14
MINDS-14 contains recordings of people asking an e-banking system questions in several languages and dialects, and has the intent_class for each recording. 
We can classify the recordings by intent of the call.

```python
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
```
We’ll load the checkpoint "anton-l/xtreme_s_xlsr_300m_minds14", which is an XLS-R model fine-tuned on MINDS-14 for approximately 50 epochs. 
It achieves 90% accuracy over all languages from MINDS-14 on the evaluation set.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)


# pass a sample to the classification pipeline to make a prediction:
classifier(minds[0]["audio"])
```
Great! We’ve identified that the intent of the call was paying a bill, with probability 96%. You can imagine this kind of keyword spotting system being used as the first stage of an automated call centre, where we want to categorise incoming customer calls based on their query and offer them contextualised support accordingly.


### Speech Commands
Speech Commands is a dataset of spoken words designed to evaluate audio classification models on simple command words. 
The dataset consists of 15 classes of keywords, a class for silence, and an unknown class to include the false positive. 
The 15 keywords are single words that would typically be used in on-device settings to control basic tasks or launch other processes.


A similar model is running continuously on your mobile phone. Here, instead of having single command words, we have ‘wake words’ specific to your device, such as “Hey Google” or “Hey Siri”. When the audio classification model detects these wake words, it triggers your phone to start listening to the microphone and transcribe your speech using a speech recognition model.

The audio classification model is much smaller and lighter than the speech recognition model, often only several millions of parameters compared to several hundred millions for speech recognition.

Let’s load a sample of the Speech Commands dataset using streaming mode:
```python
speech_commands = load_dataset(
    "speech_commands", "v0.02", split="validation", streaming=True
)
sample = next(iter(speech_commands))
```
We’ll load an official Audio Spectrogram Transformer checkpoint fine-tuned on the Speech Commands dataset, under the namespace "MIT/ast-finetuned-speech-commands-v2":

```python
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2"
)
classifier(sample["audio"].copy())
```

Output:

```
[{'score': 0.9999892711639404, 'label': 'backward'},
 {'score': 1.7504888774055871e-06, 'label': 'happy'},
 {'score': 6.703040185129794e-07, 'label': 'follow'},
 {'score': 5.805884484288981e-07, 'label': 'stop'},
 {'score': 5.614546694232558e-07, 'label': 'up'}]
```

### Language Identification
Language identification (LID) is the task of identifying the language spoken in an audio sample from a list of candidate languages. 
LID can form an important part in many speech pipelines. 
For example, given an audio sample in an unknown language, an LID model can be used to categorise the language(s) spoken in the audio sample, and then select an appropriate speech recognition model trained on that language to transcribe the audio.


##### FLEURS
FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a dataset for evaluating speech recognition systems in 102 languages, including many that are classified as ‘low-resource’. 

Let’s load up a sample from the validation split of the FLEURS dataset using streaming mode:
```python
fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
sample = next(iter(fleurs))
```
Great! Now we can load our audio classification model. For this, we’ll use a version of Whisper fine-tuned on the FLEURS dataset, which is currently the most performant LID model on the Hub:
```python
classifier = pipeline(
    "audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id"
)
# pass the audio through our classifier and generate a prediction:
classifier(sample["audio"])
```

Output:
```
[{'score': 0.9999330043792725, 'label': 'Afrikaans'},
 {'score': 7.093023668858223e-06, 'label': 'Northern-Sotho'},
 {'score': 4.269149485480739e-06, 'label': 'Icelandic'},
 {'score': 3.2661141631251667e-06, 'label': 'Danish'},
 {'score': 3.2580724109720904e-06, 'label': 'Cantonese Chinese'}]
```

### Zero-Shot Audio Classification

In the traditional paradigm for audio classification, the model predicts a class label from a pre-defined set of possible classes. 
This poses a barrier to using pre-trained models for audio classification, since the label set of the pre-trained model must match that of the downstream task.

Zero-shot audio classification is a method for taking a pre-trained audio classification model trained on a set of labelled examples and enabling it to be able to classify new examples from previously unseen classes. 
Let’s take a look at how we can achieve this!

Currently, 🤗 Transformers supports one kind of model for zero-shot audio classification: the CLAP model. 
CLAP is a transformer-based model that takes both audio and text as inputs, and computes the similarity between the two. 
If we pass a text input that strongly correlates with an audio input, we’ll get a high similarity score. 
Conversely, passing a text input that is completely unrelated to the audio input will return a low similarity.


We can use this similarity prediction for zero-shot audio classification by passing one audio input to the model and multiple candidate labels. The model will return a similarity score for each of the candidate labels, and we can pick the one that has the highest score as our prediction.

Let’s take an example where we use one audio input from the Environmental Speech Challenge (ESC) dataset:
```python
dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
audio_sample = next(iter(dataset))["audio"]["array"]
```

We then define our candidate labels, which form the set of possible classification labels. 
The model will return a classification probability for each of the labels we define. 
This means we need to know a-priori the set of possible labels in our classification problem, such that the correct label is contained within the set and is thus assigned a valid probability score.

```
candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]

# run both through the model to find the candidate label that is most similar to the audio input:
classifier = pipeline(
    task="zero-shot-audio-classification", model="laion/clap-htsat-unfused"
)
classifier(audio_sample, candidate_labels=candidate_labels)
```
Output:
```
[{'score': 0.9997242093086243, 'label': 'Sound of a dog'}, {'score': 0.0002758323971647769, 'label': 'Sound of vacuum cleaner'}]
```

Note: Why don’t we use the zero-shot audio classification pipeline for **all** audio classification tasks? 
CLAP is pre-trained on generic audio classification data, similar to the environmental sounds in the ESC dataset, rather than specifically speech data, like we had in the LID task. 
If you gave it speech in English and speech in Spanish, CLAP would know that both examples were speech data 🗣️ 
But it wouldn’t be able to differentiate between the languages in the same way a dedicated LID model is able to.



