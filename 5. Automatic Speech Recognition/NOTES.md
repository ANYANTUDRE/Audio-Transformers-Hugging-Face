# Introduction

Speech recognition, also known as automatic speech recognition (ASR) or speech-to-text (STT), is one of the most popular and exciting spoken language processing tasks. It’s used in a wide range of applications, including dictation, voice assistants, video captioning and meeting transcriptions.

![]()

ASR model broadly fall into one of two categories:
- **Connectionist Temporal Classification (CTC):** encoder-only models with a linear classification (CTC) head on top (e.g., Wav2Vec2, HuBERT and XLSR).
- **Sequence-to-sequence (Seq2Seq):** encoder-decoder models, with a cross-attention mechanism between the encoder and decoder


# I. Pre-trained models for automatic speech recognition
Prior to 2022, CTC was the more popular of the two architectures achieving breakthoughs in the pre-training / fine-tuning paradigm for speech.
Big corporations, such as Meta and Microsoft, pre-trained the encoder on vast amounts of unlabelled audio data for many days or weeks. Users could then take a pre-trained checkpoint, and fine-tune it with a CTC head on as little as 10 minutes of labelled speech data to achieve strong performance on a downstream speech recognition task.


However, CTC models have their shortcomings. Appending a simple linear layer to an encoder gives a small, fast overall model, but can be prone to phonetic spelling errors. We’ll demonstrate this for the Wav2Vec2 model below.


### Probing CTC Models

Let’s load a small excerpt of the LibriSpeech ASR dataset to demonstrate Wav2Vec2’s speech transcription capabilities:
```python
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
dataset

# inspect an audio sample
from IPython.display import Audio

sample = dataset[2]

print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```
Output:
```
HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
```

Having chosen a data sample, we now load a fine-tuned checkpoint into the pipeline(). For this, we’ll use the official Wav2Vec2 base checkpoint fine-tuned on 100 hours of LibriSpeech data:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
```

Next, we’ll take an example from the dataset and pass its raw data to the pipeline. Since the pipeline consumes any dictionary that we pass it (meaning it cannot be re-used), we’ll pass a copy of the data. This way, we can safely re-use the same audio sample in the following examples:

```python
pipe(sample["audio"].copy())
```
Output:
```
{"text": "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAUS AND ROSE BEEF LOOMING BEFORE US SIMALYIS DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"}
```

We can see that the Wav2Vec2 model does a pretty good job at transcribing this sample - at a first glance it looks generally correct. Let’s put the target and prediction side-by-side and highlight the differences:

```
Target:      HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Prediction:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH **CHRISTMAUS** AND **ROSE** BEEF LOOMING BEFORE US **SIMALYIS** DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
```
Comparing the target text to the predicted transcription, we can see that all words sound correct, but some are not spelled accurately.


This highlights the shortcoming of a CTC model. A CTC model is essentially an ‘acoustic-only’ model: it consists of an encoder which forms hidden-state representations from the audio inputs, and a linear layer which maps the hidden-states to characters:

This means that the system almost entirely bases its prediction on the acoustic input it was given (the phonetic sounds of the audio), and so has a tendency to transcribe the audio in a phonetic way (e.g. CHRISTMAUS). It gives less importance to the language modelling context of previous and successive letters, and so is prone to phonetic spelling errors. A more intelligent model would identify that CHRISTMAUS is not a valid word in the English vocabulary, and correct it to CHRISTMAS when making its predictions. We’re also missing two big features in our prediction - casing and punctuation - which limits the usefulness of the model’s transcriptions to real-world applications.


### Graduation to Seq2Seq

Cue Seq2Seq models! 
Seq2Seq models are formed of an encoder and decoder linked via a cross-attention mechanism. 
- The encoder plays the same role as before, computing hidden-state representations of the audio inputs,
- while the decoder plays the role of a language model. The decoder processes the entire sequence of hidden-state representations from the encoder and generates the corresponding text transcriptions.
With global context of the audio input, the decoder is able to use language modelling context as it makes its predictions, correcting for spelling mistakes on-the-fly and thus circumventing the issue of phonetic predictions.

There are two downsides to Seq2Seq models:

- They are inherently slower at decoding, since the decoding process happens one step at a time, rather than all at once
- They are more data hungry, requiring significantly more training data to reach convergence


In particular, the need for large amounts of training data has been a bottleneck in the advancement of Seq2Seq architectures for speech. Labelled speech data is difficult to come by, with the largest annotated datasets at the time clocking in at just 10,000 hours. This all changed in 2022 upon the release of Whisper. Whisper is a pre-trained model for speech recognition published in September 2022.

Unlike its CTC predecessors, which were pre-trained entirely on un-labelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise.


This is an order of magnitude more data than the un-labelled audio data used to train Wav2Vec 2.0 (60,000 hours). What is more, 117,000 hours of this pre-training data is multilingual (or “non-English”) data. This results in checkpoints that can be applied to over 96 languages, many of which are considered low-resource, meaning the language lacks a large corpus of data suitable for training.

When scaled to 680,000 hours of labelled pre-training data, Whisper models demonstrate a strong ability to generalise to many datasets and domains. The pre-trained checkpoints achieve competitive results to state-of-the-art pipe systems, with near 3% word error rate (WER) on the test-clean subset of LibriSpeech pipe and a new state-of-the-art on TED-LIUM with 4.7% WER


Of particular importance is Whisper’s ability to handle long-form audio samples, its robustness to input noise and ability to predict cased and punctuated transcriptions. This makes it a viable candidate for real-world speech recognition systems.

In many situations, the pre-trained Whisper checkpoints are extremely performant and give great results, thus we encourage you to try using the pre-trained checkpoints as a first step to solving any speech recognition problem. Through fine-tuning, the pre-trained checkpoints can be adapted for specific datasets and languages to further improve upon these results.



Let’s load the Whisper Base checkpoint, which is of comparable size to the Wav2Vec2 checkpoint we used previously. Preempting our move to multilingual speech recognition, we’ll load the multilingual variant of the base checkpoint. We’ll also load the model on the GPU if available, or CPU otherwise. The pipeline() will subsequently take care of moving all inputs / outputs from the CPU to the GPU as required:

```python
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
# transcribe the audio with max_new_tokens, the maximum number of tokens to generate
pipe(sample["audio"], max_new_tokens=256)
```
Output:
```
{'text': ' He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.'}
```
Easy enough! The first thing you’ll notice is the presence of both casing and punctuation. Immediately this makes the transcription easier to read compared to the un-cased and un-punctuated transcription from Wav2Vec2. Let’s put the transcription side-by-side with the target:
```
Target:     HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Prediction: He tells us that at this festive season of the year, with **Christmas** and **roast** beef looming before us, **similarly** is drawn from eating and its results occur most readily to the mind.
```

Whisper has done a great job at correcting the phonetic errors we saw from Wav2Vec2 - both Christmas and roast are spelled correctly. We see that the model still struggles with SIMILES, being incorrectly transcribed as similarly, but this time the prediction is a valid word from the English vocabulary. Using a larger Whisper checkpoint can help further reduce transcription errors, at the expense of requiring more compute and a longer transcription time.


We’ve been promised a model that can handle 96 languages, so lets leave English speech recognition for now and go global 🌎! The Multilingual LibriSpeech (MLS) dataset is the multilingual equivalent of the LibriSpeech dataset, with labelled audio data in six languages. We’ll load one sample from the Spanish split of the MLS dataset, making use of streaming mode so that we don’t have to download the entire dataset:

```python
dataset = load_dataset(
    "facebook/multilingual_librispeech", "spanish", split="validation", streaming=True
)
sample = next(iter(dataset))
```
Again, we’ll inspect the text transcription and take a listen to the audio segment:

```python
print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```
Output:
```
entonces te delelitarás en jehová y yo te haré subir sobre las alturas de la tierra y te daré á comer la heredad de jacob tu padre porque la boca de jehová lo ha hablado
```
This is the target text that we’re aiming for with our Whisper transcription. Although we now know that we can probably do better this, since our model is also going to predict punctuation and casing, neither of which are present in the reference. Let’s forward the audio sample to the pipeline to get our text prediction. One thing to note is that the pipeline consumes the dictionary of audio inputs that we input, meaning the dictionary can’t be re-used. To circumvent this, we’ll pass a copy of the audio sample, so that we can re-use the same audio sample in the proceeding code examples:

```python
pipe(sample["audio"].copy(), max_new_tokens=256, generate_kwargs={"task": "transcribe"})
```
Output:
```
{'text': ' Entonces te deleitarás en Jehová y yo te haré subir sobre las alturas de la tierra y te daré a comer la heredad de Jacob tu padre porque la boca de Jehová lo ha hablado.'}
```

Great - this looks very similar to our reference text (arguably better since it has punctuation and casing!). You’ll notice that we forwarded the "task" as a generate key-word argument (generate kwarg). Setting the "task" to "transcribe" forces Whisper to perform the task of speech recognition, where the audio is transcribed in the same language that the speech was spoken in. Whisper is also capable of performing the closely related task of speech translation, where the audio in Spanish can be translated to text in English. To achieve this, we set the "task" to "translate":

```python
pipe(sample["audio"], max_new_tokens=256, generate_kwargs={"task": "translate"})
```
Output:
```
{'text': ' So you will choose in Jehovah and I will raise you on the heights of the earth and I will give you the honor of Jacob to your father because the voice of Jehovah has spoken to you.'}
```

### Long-Form Transcription and Timestamps
So far, we’ve focussed on transcribing short audio samples of less than 30 seconds. We mentioned that one of the appeals of Whisper was its ability to work on long audio samples. We’ll tackle this task here!

Let’s create a long audio file by concatenating sequential samples from the MLS dataset. Since the MLS dataset is curated by splitting long audiobook recordings into shorter segments, concatenating samples is one way of reconstructing longer audiobook passages. Consequently, the resulting audio should be coherent across the entire sample.

We’ll set our target audio length to 5 minutes, and stop concatenating samples once we hit this value:

```python
import numpy as np

target_length_in_m = 5

# convert from minutes to seconds (* 60) to num samples (* sampling rate)
sampling_rate = pipe.feature_extractor.sampling_rate
target_length_in_samples = target_length_in_m * 60 * sampling_rate

# iterate over our streaming dataset, concatenating samples until we hit our target
long_audio = []
for sample in dataset:
    long_audio.extend(sample["audio"]["array"])
    if len(long_audio) > target_length_in_samples:
        break

long_audio = np.asarray(long_audio)

# how did we do?
seconds = len(long_audio) / 16000
minutes, seconds = divmod(seconds, 60)
print(f"Length of audio sample is {minutes} minutes {seconds:.2f} seconds")
```
Output:

```
Length of audio sample is 5.0 minutes 17.22 seconds
```
Alright! 5 minutes and 17 seconds of audio to transcribe. There are two problems with forwarding this long audio sample directly to the model:

- Whisper is inherently designed to work with 30 second samples: anything shorter than 30s is padded to 30s with silence, anything longer than 30s is truncated to 30s by cutting of the extra audio, so if we pass our audio directly we’ll only get the transcription for the first 30s
- Memory in a transformer network scales with the sequence length squared: doubling the input length quadruples the memory requirement, so passing super long audio files is bound to lead to an out-of-memory (OOM) error

The way long-form transcription works in 🤗 Transformers is by chunking the input audio into smaller, more manageable segments. Each segment has a small amount of overlap with the previous one. This allows us to accurately stitch the segments back together at the boundaries, since we can find the overlap between segments and merge the transcriptions accordingly:

![]()


The advantage of chunking:


To activate long-form transcriptions, we have to add one additional argument when we call the pipeline. This argument, chunk_length_s, controls the length of the chunked segments in seconds. For Whisper, 30 second chunks are optimal, since this matches the input length Whisper expects.

To activate batching, we need to pass the argument batch_size to the pipeline. Putting it all together, we can transcribe the long audio sample with chunking and batching as follows:

```python
pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
)
```
Output:

```python
{'text': ' Entonces te deleitarás en Jehová, y yo te haré subir sobre las alturas de la tierra, y te daré a comer la
heredad de Jacob tu padre, porque la boca de Jehová lo ha hablado. nosotros curados. Todos nosotros nos descarriamos
como bejas, cada cual se apartó por su camino, mas Jehová cargó en él el pecado de todos nosotros...
```

Whisper is also able to predict segment-level timestamps for the audio data. These timestamps indicate the start and end time for a short passage of audio, and are particularly useful for aligning a transcription with the input audio. Suppose we want to provide closed captions for a video - we need these timestamps to know which part of the transcription corresponds to a certain segment of video, in order to display the correct transcription for that time.

Activating timestamp prediction is straightforward, we just need to set the argument return_timestamps=True. Timestamps are compatible with both the chunking and batching methods we used previously, so we can simply append the timestamp argument to our previous call:

```python
pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
)["chunks"]
```

Output:
```python
[{'timestamp': (0.0, 26.4),
  'text': ' Entonces te deleitarás en Jehová, y yo te haré subir sobre las alturas de la tierra, y te daré a comer la heredad de Jacob tu padre, porque la boca de Jehová lo ha hablado. nosotros curados. Todos nosotros nos descarriamos como bejas, cada cual se apartó por su camino,'},
 {'timestamp': (26.4, 32.48),
  'text': ' mas Jehová cargó en él el pecado de todos nosotros. No es que partas tu pan con el'},
 {'timestamp': (32.48, 38.4),
  'text': ' hambriento y a los hombres herrantes metas en casa, que cuando vieres al desnudo lo cubras y no'},
 ...
```
And voila! We have our predicted text as well as corresponding timestamps.

### Summary
Whisper is a strong pre-trained model for speech recognition and translation. Compared to Wav2Vec2, it has higher transcription accuracy, with outputs that contain punctuation and casing. It can be used to transcribe speech in English as well as 96 other languages, both on short audio segments and longer ones through chunking. These attributes make it a viable model for many speech recognition and translation tasks without the need for fine-tuning. The pipeline() method provides an easy way of running inference in one-line API calls with control over the generated predictions.

While the Whisper model performs extremely well on many high-resource languages, it has lower transcription and translation accuracy on low-resource languages, i.e. those with less readily available training data.


# II. Choosing a dataset

As with any machine learning problem, our model is only as good as the data that we train it on. Speech recognition datasets vary considerably in how they are curated and the domains that they cover. To pick the right dataset, we need to match our criteria with the features that a dataset offers.

### Features of speech datasets

##### 1. Number of hours
Simply put, the number of training hours indicates how large the dataset is. It’s analogous to the number of training examples in an NLP dataset. However, bigger datasets aren’t necessarily better. If we want a model that generalises well, we want a diverse dataset with lots of different speakers, domains and speaking styles.

##### 2. Domain
The domain entails where the data was sourced from, whether it be audiobooks, podcasts, YouTube or financial meetings. Each domain has a different distribution of data. For example, audiobooks are recorded in high-quality studio conditions (with no background noise) and text that is taken from written literature. Whereas for YouTube, the audio likely contains more background noise and a more informal style of speech.

We need to match our domain to the conditions we anticipate at inference time. For instance, if we train our model on audiobooks, we can’t expect it to perform well in noisy environments.


##### 3. Speaking style

The speaking style falls into one of two categories:

- Narrated: read from a script (tends to be spoken articulately and without any errors)
- Spontaneous: un-scripted, conversational speech (a more colloquial style of speech, with the inclusion of repetitions, hesitations and false-starts)

##### 4. Transcription style

The transcription style refers to whether the target text has punctuation, casing or both. If we want a system to generate fully formatted text that could be used for a publication or meeting transcription, we require training data with punctuation and casing. If we just require the spoken words in an un-formatted structure, neither punctuation nor casing are necessary. In this case, we can either pick a dataset without punctuation or casing, or pick one that has punctuation and casing and then subsequently remove them from the target text through pre-processing.

### A summary of datasets on the Hub





### Common Voice 13



# III. Evaluation metrics for ASR

When assessing speech recognition systems, we compare the system’s predictions to the target text transcriptions, annotating any errors that are present. We categorise these errors into one of three categories:
- 1. Substitutions (S): where we transcribe the wrong word in our prediction (“sit” instead of “sat”)
- 2. Insertions (I): where we add an extra word in our prediction
- 3. Deletions (D): where we remove a word in our prediction

These error categories are the same for all speech recognition metrics. What differs is the level at which we compute these errors: we can either compute them on the word level or on the character level.

We’ll use a running example for each of the metric definitions. Here, we have a ground truth or reference text sequence:
```
reference = "the cat sat on the mat"
```
And a predicted sequence from the speech recognition system that we’re trying to assess:
```
prediction = "the cat sit on the"
```

We can see that the prediction is pretty close, but some words are not quite right. We’ll evaluate this prediction against the reference for the three most popular speech recognition metrics and see what sort of numbers we get for each.

### Word Error Rate
The word error rate (WER) metric is the ‘de facto’ metric for speech recognition. It calculates substitutions, insertions and deletions on the word level. This means errors are annotated on a word-by-word basis. Take our example:

![]()

Here, we have:

- 1 substitution (“sit” instead of “sat”)
- 0 insertions
- 1 deletion (“mat” is missing)

This gives 2 errors in total. To get our error rate, we divide the number of errors by the total number of words in our reference (N), which for this example is 6:

![]()


Alright! So we have a WER of 0.333, or 33.3%. Notice how the word “sit” only has one character that is wrong, but the entire word is marked incorrect. This is a defining feature of the WER: spelling errors are penalised heavily, no matter how minor they are.

The WER is defined such that lower is better: a lower WER means there are fewer errors in our prediction, so a perfect speech recognition system would have a WER of zero (no errors).

Let’s see how we can compute the WER using 🤗 Evaluate. We’ll need two packages to compute our WER metric: 🤗 Evaluate for the API interface, and JIWER to do the heavy lifting of running the calculation:

```python
pip install --upgrade evaluate jiwer
```

Great! We can now load up the WER metric and compute the figure for our example:
```python
from evaluate import load

wer_metric = load("wer")

wer = wer_metric.compute(references=[reference], predictions=[prediction])

print(wer)
```
Print Output:
```python
0.3333333333333333
```

Now, here’s something that’s quite confusing… What do you think the upper limit of the WER is? You would expect it to be 1 or 100% right? Nuh uh! Since the WER is the ratio of errors to number of words (N), there is no upper limit on the WER! Let’s take an example were we predict 10 words and the target only has 2 words. If all of our predictions were wrong (10 errors), we’d have a WER of 10 / 2 = 5, or 500%! This is something to bear in mind if you train an ASR system and see a WER of over 100%. Although if you’re seeing this, something has likely gone wrong… 😅


### Word Accuracy

We can flip the WER around to give us a metric where higher is better. Rather than measuring the word error rate, we can measure the word accuracy (WAcc) of our system:
![]()


### Character Error Rate
It seems a bit unfair that we marked the entire word for “sit” wrong when in fact only one letter was incorrect. That’s because we were evaluating our system on the word level, thereby annotating errors on a word-by-word basis. The character error rate (CER) assesses systems on the character level. This means we divide up our words into their individual characters, and annotate errors on a character-by-character basis:

![]()

We can see now that for the word “sit”, the “s” and “t” are marked as correct. It’s only the “i” which is labelled as a substitution error (S). Thus, we reward our system for the partially correct prediction 🤝

In our example, we have 1 character substitution, 0 insertions, and 3 deletions. In total, we have 14 characters. So, our CER is:

![]()


### Which metric should I use?
In general, the WER is used far more than the CER for assessing speech systems. This is because the WER requires systems to have greater understanding of the context of the predictions. In our example, “sit” is in the wrong tense. A system that understands the relationship between the verb and tense of the sentence would have predicted the correct verb tense of “sat”. We want to encourage this level of understanding from our speech systems. So although the WER is less forgiving than the CER, it’s also more conducive to the kinds of intelligible systems we want to develop. Therefore, we typically use the WER and would encourage you to as well! 

However, there are circumstances where it is not possible to use the WER. Certain languages, such as Mandarin and Japanese, have no notion of ‘words’, and so the WER is meaningless. Here, we revert to using the CER.

### Normalisation
If we train an ASR model on data with punctuation and casing, it will learn to predict casing and punctuation in its transcriptions. This is great when we want to use our model for actual speech recognition applications, such as transcribing meetings or dictation, since the predicted transcriptions will be fully formatted with casing and punctuation, a style referred to as orthographic.

However, we also have the option of normalising the dataset to remove any casing and punctuation. Normalising the dataset makes the speech recognition task easier: the model no longer needs to distinguish between upper and lower case characters, or have to predict punctuation from the audio data alone (e.g. what sound does a semi-colon make?). Because of this, the word error rates are naturally lower (meaning the results are better). The Whisper paper demonstrates the drastic effect that normalising transcriptions can have on WER results (c.f. Section 4.4 of the Whisper paper). While we get lower WERs, the model isn’t necessarily better for production. The lack of casing and punctuation makes the predicted text from the model significantly harder to read. Take the example from the previous section, where we ran Wav2Vec2 and Whisper on the same audio sample from the LibriSpeech dataset. The Wav2Vec2 model predicts neither punctuation nor casing, whereas Whisper predicts both. Comparing the transcriptions side-by-side, we see that the Whisper transcription is far easier to read:

```
Wav2Vec2:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAUS AND ROSE BEEF LOOMING BEFORE US SIMALYIS DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Whisper:   He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.
```
The Whisper transcription is orthographic and thus ready to go - it’s formatted as we’d expect for a meeting transcription or dictation script with both punctuation and casing. On the contrary, we would need to use additional post-processing to restore punctuation and casing in our Wav2Vec2 predictions if we wanted to use it for downstream applications.

There is a happy medium between normalising and not normalising: we can train our systems on orthographic transcriptions, and then normalise the predictions and targets before computing the WER. This way, we train our systems to predict fully formatted text, but also benefit from the WER improvements we get by normalising the transcriptions.

The Whisper model was released with a normaliser that effectively handles the normalisation of casing, punctuation and number formatting among others. Let’s apply the normaliser to the Whisper transcriptions to demonstrate how we can normalise them:

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

prediction = " He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind."
normalized_prediction = normalizer(prediction)

normalized_prediction
```

Output:
```
' he tells us that at this festive season of the year with christmas and roast beef looming before us similarly is drawn from eating and its results occur most readily to the mind '
```
Great! We can see that the text has been fully lower-cased and all punctuation removed. Let’s now define the reference transcription and then compute the normalised WER between the reference and prediction:


```python
reference = "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"
normalized_referece = normalizer(reference)

wer = wer_metric.compute(
    references=[normalized_referece], predictions=[normalized_prediction]
)
wer
```
Output:

```python
0.0625
```
6.25% - that’s about what we’d expect for the Whisper base model on the LibriSpeech validation set. As we see here, we’ve predicted an orthographic transcription, but benefited from the WER boost obtained by normalising the reference and prediction prior to computing the WER.

The choice of how you normalise the transcriptions is ultimately down to your needs. We recommend training on orthographic text and evaluating on normalised text to get the best of both worlds.


### Putting it all together
We’re going to set ourselves up for the next section on fine-tuning by evaluating the pre-trained Whisper model on the Common Voice 13 Dhivehi test set. We’ll use the WER number we get as a baseline for our fine-tuning run, or a target number that we’ll try and beat 🥊

First, we’ll load the pre-trained Whisper model using the pipeline() class. This process will be extremely familiar by now! The only new thing we’ll do is load the model in half-precision (float16) if running on a GPU - this will speed up inference at almost no cost to WER accuracy.


```python
from transformers import pipeline
import torch

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=torch_dtype,
    device=device,
)
```
Next, we’ll load the Dhivehi test split of Common Voice 13. 


```python
#  link our HF account to our notebook to have access to the gated dataset
from huggingface_hub import notebook_login

notebook_login()
```

```python
from datasets import load_dataset

# downloading the Common Voice dataset
common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)
```
Evaluating over an entire dataset can be done in much the same way as over a single example - all we have to do is loop over the input audios, rather than inferring just a single sample. To do this, we first transform our dataset into a KeyDataset. All this does is pick out the particular dataset column that we want to forward to the model (in our case, that’s the "audio" column), ignoring the rest (like the target transcriptions, which we don’t want to use for inference). We then iterate over this transformed datasets, appending the model outputs to a list to save the predictions. The following code cell will take approximately five minutes if running on a GPU with half-precision, peaking at 12GB memory:

```python
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

# run streamed inference
for prediction in tqdm(
    pipe(
        KeyDataset(common_voice_test, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(common_voice_test),
):
    all_predictions.append(prediction["text"])
```
And finally, we can compute the WER. Let’s first compute the orthographic WER, i.e. the WER without any post-processing:

```python
from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
wer_ortho
```

Output:
```python
167.29577268612022
```

Okay… 167% essentially means our model is outputting garbage 😜 Not to worry, it’ll be our aim to improve this by fine-tuning the model on the Dhivehi training set!

Next, we’ll evaluate the normalised WER, i.e. the WER with normalisation post-processing. We have to filter out samples that would be empty after normalisation, as otherwise the total number of words in our reference (N) would be zero, which would give a division by zero error in our calculation:

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# compute normalised WER
all_predictions_norm = [normalizer(pred) for pred in all_predictions]
all_references_norm = [normalizer(label) for label in common_voice_test["sentence"]]

# filtering step to only evaluate the samples that correspond to non-zero references
all_predictions_norm = [
    all_predictions_norm[i]
    for i in range(len(all_predictions_norm))
    if len(all_references_norm[i]) > 0
]
all_references_norm = [
    all_references_norm[i]
    for i in range(len(all_references_norm))
    if len(all_references_norm[i]) > 0
]

wer = 100 * wer_metric.compute(
    references=all_references_norm, predictions=all_predictions_norm
)

wer
```
Output:

```python
125.69809089960707
```
Again we see the drastic reduction in WER we achieve by normalising our references and predictions: the baseline model achieves an orthographic test WER of 168%, while the normalised WER is 126%.

Right then! These are the numbers that we want to try and beat when we fine-tune the model, in order to improve the Whisper model for Dhivehi speech recognition.



# Fine-tuning the ASR model

We’ll cover a step-by-step guide on fine-tuning **Whisper small** for ASR on the Common Voice 13 dataset. 

### 1. Prepare Environment

### 2. Load Dataset
Since Dhivehi is very low-resource, we’ll combine the train and validation splits to give approximately seven hours of training data. We’ll use the three hours of test data as our held-out test set:

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)

print(common_voice)
```
Output:
```python
DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 4904
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 2212
    })
})
```

We only consider the input audio and transcribed text for fine-tuning, discarding the additional metadata information:
```python
common_voice = common_voice.select_columns(["audio", "sentence"])
```

### 3. Feature Extractor, Tokenizer and Processor
The ASR pipeline can be de-composed into three stages:

- 1. **The feature extractor** which pre-processes the raw audio-inputs to log-mel spectrograms
-  2. **The model** which performs the sequence-to-sequence mapping
-  3. **The tokenizer** which post-processes the predicted tokens to text

In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer, called WhisperFeatureExtractor and WhisperTokenizer respectively. To make our lives simple, these two objects are wrapped under a single class, called the WhisperProcessor.

When performing multilingual fine-tuning, we need to set the "language" (source audio language) and "task" ("transcribe" for ASR or "translate" for speech translation) when instantiating the processor.

```python
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

TO_LANGUAGE_CODE
```

If you scroll through this list, you’ll notice that many languages are present, but Dhivehi is one of few that is not! This means that Whisper was not pre-trained on Dhivehi. 

However, this doesn’t mean that we can’t fine tune Whisper on it. **In doing so, we’ll be teaching Whisper a new language**, one that the pre-trained checkpoint does not support. That’s pretty cool, right!

When you fine-tune it on a new language, Whisper does a good job at leveraging its knowledge of the other 96 languages it’s pre-trained on. Largely speaking, all modern languages will be linguistically similar to at least one of the 96 languages Whisper already knows, so we’ll fall under this **paradigm of cross-lingual knowledge representation.**

What we need to do to fine-tune Whisper on a new language is find the language most similar that Whisper was pre-trained on. The Wikipedia article for Dhivehi states that Dhivehi is closely related to the Sinhalese language of Sri Lanka. If we check the language codes again, we can see that Sinhalese is present in the Whisper language set, so we can safely set our language argument to "sinhalese".

```python
# load our processor from the pre-trained checkpoint
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sinhalese", task="transcribe"
)
```

### Pre-Process the Data
Let’s have a look at the dataset features. Pay particular attention to the "audio" column - this details the sampling rate of our audio inputs:

```python
common_voice["train"].features
```
Output:
```python
{'audio': Audio(sampling_rate=48000, mono=True, decode=True, id=None),
 'sentence': Value(dtype='string', id=None)}
```
Since our input audio is sampled at 48kHz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

```python
from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

Now we can write a function to prepare our data ready for the model:
1. We load and resample the audio data on a sample-by-sample basis by calling sample["audio"].
2. We use the feature extractor to compute the log-mel spectrogram input features from our 1-dimensional audio array.
3. We encode the transcriptions to label ids through the use of the tokenizer.

```python
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example
```
We can apply the data preparation function to all of our training examples using 🤗 Datasets’ .map method ( removing the columns from the raw training data - the audio and text)

```python
common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)
```
Finally, we filter any training data with audio samples longer than 30s. These samples would otherwise be truncated by the Whisper feature-extractor which could affect the stability of training.

```python
max_input_length = 30.0

# function that returns True for samples that are less than 30s, and False for those that are longer
def is_audio_in_length_range(length):
    return length < max_input_length

# apply our filter function to all samples of our training dataset through
common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
```

### Training and Evaluation
Now that we’ve prepared our data, we’re ready to dive into the training pipeline. The 🤗 Trainer will do much of the heavy lifting for us. All we have to do is:

- Define a **data collator** that takes our pre-processed data and prepares PyTorch tensors ready for the model.
- **Evaluation metrics** to evaluate the model using the WER metric.
- Load a **pre-trained checkpoint** and configure it correctly for training.
- Define the **training arguments** that will be used by the 🤗 Trainer in constructing the training schedule.

Once we’ve fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it to transcribe speech in Dhivehi.

##### Define a Data Collator

The data collator for a sequence-to-sequence speech model is unique in the sense that it treats the input_features and labels independently: the input_features must be handled by the feature extractor and the labels by the tokenizer.

- The input_features are already padded to 30s and converted to a log-Mel spectrogram of fixed dimension, so all we have to do is convert them to batched PyTorch tensors. We do this using the feature extractor’s .pad method with return_tensors=pt. Note that no additional padding is applied here since the inputs are of fixed dimension, the input_features are simply converted to PyTorch tensors.

- On the other hand, the labels are un-padded. We first pad the sequences to the maximum length in the batch using the tokenizer’s .pad method. The padding tokens are then replaced by -100 so that these tokens are not taken into account when computing the loss. We then cut the start of transcript token from the beginning of the label sequence as we append it later during training.

```python
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

#  initialise the data collator we’ve just defined
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

##### Evaluation Metrics
We’ll use the Word Error Rate (WER) metric, the ‘de-facto’ metric for assessing ASR systems.
```python
# load the WER metric from 🤗 Evaluate
import evaluate
metric = evaluate.load("wer")
```
We then simply have to define a function that takes our model predictions and returns the WER metric. This function, called compute_metrics, first replaces -100 with the pad_token_id in the label_ids (undoing the step we applied in the data collator to ignore padded tokens correctly in the loss). It then decodes the predicted and label ids to strings. Finally, it computes the WER between the predictions and reference labels. Here, we have the option of evaluating with the ‘normalised’ transcriptions and predictions, which have punctuation and casing removed. We recommend you follow this to benefit from the WER improvement obtained by normalising the transcriptions.

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
```

##### Load a Pre-Trained Checkpoint


```python
from transformers import WhisperForConditionalGeneration

#  load the pre-trained Whisper small checkpoint
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

We’ll set use_cache to False for training since we’re using gradient checkpointing and the two are incompatible. We’ll also override two generation arguments to control the behaviour of the model during inference: we’ll force the language and task tokens during generation by setting the language and task arguments, and also re-enable cache for generation to speed-up inference time:

```python
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="sinhalese", task="transcribe", use_cache=True
)
```

### Define the Training Configuration
In the final step, we define all the parameters related to training. Here, we set the number of training steps to 500. This is enough steps to see a big WER improvement compared to the pre-trained Whisper model, while ensuring that fine-tuning can be run in approximately 45 minutes on a Google Colab free tier.

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
```
We can forward the training arguments to the 🤗 Trainer along with our model, dataset, data collator and compute_metrics function:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```

### Training
To launch training, simply execute:

```python
trainer.train()
```
Depending on your GPU, it is possible that you will encounter a CUDA "out-of-memory" error when you start training. In this case, you can reduce the per_device_train_batch_size incrementally by factors of 2 and employ gradient_accumulation_steps to compensate.

The big question is how this compares to other ASR systems. For that, we can view the autoevaluate leaderboard, a leaderboard that categorises models by language and dataset, and subsequently ranks them according to their WER.

Looking at the leaderboard, we see that our model trained for 500 steps convincingly beats the pre-trained Whisper Small checkpoint that we evaluated in the previous section. Nice job 👏

We can automatically submit our checkpoint to the leaderboard when we push the training results to the Hub - we simply have to set the appropriate key-word arguments (kwargs). You can change these values to match your dataset, language and model name accordingly:


```python
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_13_0",
    "dataset": "Common Voice 13",  # a 'pretty' name for the training dataset
    "language": "dv",
    "model_name": "Whisper Small Dv - Sanchit Gandhi",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

# upload to the Hub
trainer.push_to_hub(**kwargs)
```

While the fine-tuned model yields satisfactory results on the Common Voice 13 Dhivehi test data, it is by no means optimal. The purpose of this guide is to demonstrate how to fine-tune an ASR model using the 🤗 Trainer for multilingual speech recognition.

If you decide to train for 4000 steps, we also recommend changing the learning rate scheduler to a linear schedule (set lr_scheduler_type="linear"), as this will yield an additional performance boost over long training runs.

The results could likely be improved further by optimising the training hyperparameters, such as learning rate and dropout, and using a larger pre-trained checkpoint (medium or large)

### Sharing Your Model

You can now share this model with anyone using the link on the Hub. They can load it with the identifier "your-username/the-name-you-picked" directly into the pipeline() object. 

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="sanchit-gandhi/whisper-small-dv")
```


# Build a demo with Gradio
Now that we’ve fine-tuned a Whisper model for Dhivehi speech recognition, let’s go ahead and build a Gradio demo to showcase it to the community!

The first thing to do is load up the fine-tuned checkpoint using the pipeline() class:
```python
from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)
```

Secondly, we’ll define a function that takes the filepath for an audio input and passes it through the pipeline. 

```python
def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "sinhalese",
        },           # update with the language you've fine-tuned on
        chunk_length_s=30,     # enable chunking
        batch_size=8,
    )
    return output["text"]
```

We’ll use the Gradio blocks feature to launch two tabs on our demo: one for microphone transcription, and the other for file upload.

```python
import gradio as gr

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.outputs.Textbox(),
)
```
Finally, we launch the Gradio demo using the two blocks that we’ve just defined:

```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)
```


