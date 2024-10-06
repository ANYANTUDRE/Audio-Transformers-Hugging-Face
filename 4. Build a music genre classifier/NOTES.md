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



# II. Fine-tuning a model for music classification
In this section, we’ll present a step-by-step guide on fine-tuning an encoder-only transformer model for music classification. 

### The Dataset
To train our model, we’ll use the GTZAN dataset, which is a popular dataset of 1,000 songs for music genre classification. Each song is a 30-second clip from one of 10 genres of music, spanning disco to metal.

```python
from datasets import load_dataset

gtzan = load_dataset("marsyas/gtzan", "all")
print(gtzan)

# GTZAN doesn’t provide a predefined validation set, so we’ll have to create one ourselves.
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
print(gtzan)

# take a look at one of the audio files:
print(gtzan["train"][0])
```
Output:

Some observations:
- the audio files are represented as 1-dimensional NumPy arrays, where the value of the array represents the amplitude at that timestep
- the sampling rate is 22,050 Hz, meaning there are 22,050 amplitude values sampled per second.
-  the genre is represented as an integer, or class label

```python
# use the int2str() method of the genre feature to map these integers to human-readable names
id2label_fn = gtzan["train"].features["genre"].int2str
id2label_fn(gtzan["train"][0]["genre"])
```
Output:



### Picking a pretrained model for audio classification

To get started, let’s pick a suitable pretrained model for audio classification. In this domain, pretraining is typically carried out on large amounts of unlabeled audio data, using datasets like LibriSpeech and Voxpopuli. 
The best way to find these models on the Hugging Face Hub is to use the “Audio Classification” filter

Although models like Wav2Vec2 and HuBERT are very popular, we’ll use a model called DistilHuBERT. This is a much smaller (or distilled) version of the HuBERT model, which trains around 73% faster, yet preserves most of the performance.


### Preprocessing the data

In 🤗 Transformers, the conversion from audio to the input format is handled by the *feature extractor** of the model. 

```python
from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
```
Since the sampling rate of the model and the dataset are different, we’ll have to resample the audio file to 16,000 Hz before passing it to the feature extractor. 

```python
# obtaining the model’s sample rate from the feature extractor
sampling_rate = feature_extractor.sampling_rate
print(sampling_rate)

# resample the dataset using the cast_column() method and Audio feature from 🤗 Datasets
from datasets import Audio
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
```
We can now check the first sample of the train-split of our dataset to verify that it is indeed at 16,000 Hz. 🤗 Datasets will resample the audio file on-the-fly when we load each audio sample:

```python
gtzan["train"][0]
```

A defining feature of Wav2Vec2 and HuBERT like models is that they accept a float array corresponding to the raw waveform of the speech signal as an input. 
This is in contrast to other models, like Whisper, where we pre-process the raw audio waveform to spectrogram format.


The audio data is already in the right format to be read by the model represented as a 1-dimensional array, (1-dimensional array of continuous inputs at discrete time steps). So, what exactly does the feature extractor do? **Feature scaling**!!!

The feature extractor normalise our audio data, by rescaling each sample to zero mean and unit variance ( helping with stability and convergence during training )

We can take a look at the feature extractor in operation by applying it to our first audio sample.
```python
import numpy as np

sample = gtzan["train"][0]["audio"]

# compute the mean and variance of our raw audio data
print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

# apply the feature extractor and see what the outputs look like
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
print(f"inputs keys: {list(inputs.keys())}")

print(
    f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
)
```

Output:
```

```

Alright! Our feature extractor returns a dictionary of two arrays: input_values and attention_mask. - The input_values are the preprocessed audio inputs that we’d pass to the HuBERT model. 
- The attention_mask is used when we process a batch of audio inputs at once - it is used to tell the model where we have padded inputs of different lengths.

We can see that the mean value is now very much closer to zero, and the variance bang-on one! This is exactly the form we want our audio samples in prior to feeding them to the HuBERT model.


Great, so now we know how to process our resampled audio files, the last thing to do is define a function that we can apply to all the examples in the dataset. Since we expect the audio clips to be 30 seconds in length, we’ll also truncate any longer clips by using the max_length and truncation arguments of the feature extractor as follows:

```python
max_duration = 30.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

With this function defined, we can now apply it to the dataset using the map() method. The .map() method supports working with batches of examples, which we’ll enable by setting batched=True. 
The default batch size is 1000, but we’ll reduce it to 100 to ensure the peak RAM stays within a sensible range for Google Colab’s free tier:
```python
gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
gtzan_encoded
```
Output:
```

```

To simplify the training, we’ve removed the audio and file columns from the dataset. The input_values column contains the encoded audio files, the attention_mask a binary mask of 0/1 values that indicate where we have padded the audio input, and the genre column contains the corresponding labels (or targets). 
To enable the Trainer to process the class labels, we need to rename the genre column to label:
```python
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
```

Finally, we need to obtain the label mappings from the dataset. This mapping will take us from integer ids (e.g. 7) to human-readable class labels (e.g. "pop") and back again. In doing so, we can convert our model’s integer id prediction into human-readable format, enabling us to use the model in any downstream application. We can do this by using the int2str() method as follows:
```python
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

id2label["7"]
```

### Fine-tuning the model

To fine-tune the model, we’ll use the Trainer class from 🤗 Transformers. As we’ve seen in other chapters, the Trainer is a high-level API that is designed to handle the most common training scenarios. In this case, we’ll use the Trainer to fine-tune the model on GTZAN. To do this, we’ll first need to load a model for this task. We can do this by using the AutoModelForAudioClassification class, which will automatically add the appropriate classification head to our pretrained DistilHuBERT model. Let’s go ahead and instantiate the model:
```python
from transformers import AutoModelForAudioClassification

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
```

It's recommended to upload model checkpoints directly the Hugging Face Hub while training.
Linking the notebook to the Hub is straightforward
```python
from huggingface_hub import notebook_login

notebook_login()
```

The next step is to define the training arguments, including the batch size, gradient accumulation steps, number of training epochs and learning rate:
```python
from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)
```
The last thing we need to do is define the metrics. Since the dataset is balanced, we’ll use accuracy as our metric and load it using the 🤗 Evaluate library:

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

We’ve now got all the pieces! Let’s instantiate the Trainer and train the model:

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
```
We can automatically submit our checkpoint to the leaderboard when we push the training results to the Hub - we simply have to set the appropriate key-word arguments (kwargs). You can change these values to match your dataset, language and model name accordingly:
```python
kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

# The training results can now be uploaded to the Hub. To do so, execute the .push_to_hub command:
trainer.push_to_hub(**kwargs)
```

### Share Model

```python
from transformers import pipeline

pipe = pipeline(
    "audio-classification", model="sanchit-gandhi/distilhubert-finetuned-gtzan"
)
```

Conclusion: While we focussed on the task of music classification and the GTZAN dataset, the steps presented here apply more generally to any audio classification task - the same script can be used for spoken language audio classification tasks like keyword spotting or language identification. 

# III. Build a demo with Gradio

In this final section on audio classification, we’ll build a Gradio demo to showcase the music classification model that we just trained on the GTZAN dataset.

```python
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)
```

Secondly, we’ll define a function that takes the filepath for an audio input and passes it through the pipeline. Here, the pipeline automatically takes care of loading the audio file, resampling it to the correct sampling rate, and running inference with the model. We take the models predictions of preds and format them as a dictionary object to be displayed on the output:

```python
def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs
```

Finally, we launch the Gradio demo using the function we’ve just defined:

```python
import gradio as gr

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.outputs.Label()
)
demo.launch(debug=True)
```


