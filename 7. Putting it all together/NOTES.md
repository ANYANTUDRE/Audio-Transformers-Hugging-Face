# I. Speech-to-speech translation

**Speech-to-speech translation (STST or S2ST)** involves translating speech from one langauge into speech in a different language.

![][https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/s2st.png]

STST holds applications in the field of **multilingual communication** with two possible approaches:
- **Two stage cascaded approach**: speech translation (ST) + text-to-speech (TTS).  
Advantages:
  - straightforward, effective,
  - very data and compute efficient

![](https://github.com/ANYANTUDRE/Audio-Transformers-Hugging-Face/blob/main/img/s2st_cascaded.png)

- **Three stage approach**: ASR + MT + TTS.  
Problems:
  - error propagation through the pipeline
  - increase in latency


### Speech translation
- **Model we'll use:** [Whisper Base - 74M](https://huggingface.co/openai/whisper-base)
- **Goal:** translate from any of the 96 languages to English
- **Advantages:** 
  - multilingual
  - reasonable inference speed


```python
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)

# load an audio sample in a non-English language
from datasets import load_dataset
dataset = load_dataset("facebook/voxpopuli", "it", split="validation", streaming=True)
sample = next(iter(dataset))

# function that takes this audio input and returns the translated text
def translate(audio):
    outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    return outputs["text"]

# check that we get a sensible result from the model
translate(sample["audio"].copy())
```


### Text-to-speech
- **Model we'll use:** [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) TTS model for
- **Goal:** English text synthesis  

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# place the SpeechT5 model and vocoder on our GPU accelerator device
model.to(device)
vocoder.to(device)

# load up the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# takes a text prompt as input, and generates the corresponding speech
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    # bringing back generated speech to the CPU
    return speech.cpu()

speech = synthesise("Hey there! This is a test!")
Audio(speech, rate=16000)
```


### Creating a STST demo
**Steps to follow:**
- **Quick sanity check:** to make sure we can concatenate the two models (or rather the two functions), putting an audio sample in and getting an audio sample out.
- **Convert the synthesised speech to an int16 array** (format expected by Gradio):
  - normalise the audio array by the dynamic range of the target dtype (int16)
  - convert from the default NumPy dtype (float64) to the target dtype (int16).

```python
import numpy as np

target_dtype = np.int16
max_range = np.iinfo(target_dtype).max


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * max_range).astype(np.int16)
    return 16000, synthesised_speech

sampling_rate, synthesised_speech = speech_to_speech_translation(sample["audio"])

Audio(synthesised_speech, rate=sampling_rate)
```

Gradio demo:
```python
import gradio as gr

demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate],
                        ["Microphone", "Audio File"]
                  )

demo.launch(debug=True)
```


# Creating a voice assistant

We’ll piece together three models to build an end-to-end voice assistant called Marvin 🤖. Like Amazon’s Alexa or Apple’s Siri, Marvin is a virtual voice assistant who responds to a particular ‘wake word’, then listens out for a spoken query, and finally responds with a spoken answer.

We can break down the voice assistant pipeline into four stages, each of which requires a standalone model:

![]()

### Pipeline

##### 1. Wake word detection

Voice assistants are constantly listening to the audio inputs coming through your device’s microphone, however they only boot into action when a particular ‘wake word’ or ‘trigger word’ is spoken.

The wake word detection task is handled by a small on-device audio classification model. Only when the wake word is detected is the larger speech recognition model launched, and afterwards it is shut down again.


##### 2. Speech transcription

The next stage in the pipeline is transcribing the spoken query to text. In practice, transferring audio files from your local device to the Cloud is slow due to the large nature of audio files, so it’s more efficient to transcribe them directly using an automatic speech recognition (ASR) model on-device rather than using a model in the Cloud. The on-device model might be smaller and thus less accurate than one hosted in the Cloud, but the faster inference speed makes it worthwhile since we can run speech recognition in near real-time, our spoken audio utterance being transcribed as we say it.


##### 3. Language model query
Now that we know what the user asked, we need to generate a response! The best candidate models for this task are large language models (LLMs), since they are effectively able to understand the semantics of the text query and generate a suitable response.

Since our text query is small (just a few text tokens), and language models large (many billions of parameters), the most efficient way of running LLM inference is to send our text query from our device to an LLM running in the Cloud, generate a text response, and return the response back to the device.

##### 4. Synthesise speech
Finally, we’ll use a text-to-speech (TTS) model to synthesise the text response as spoken speech. This is done on-device, but you could feasibly run a TTS model in the Cloud, generating the audio output and transferring it back to the device.


### Wake word detection
The first stage in the voice assistant pipeline is detecting whether the wake word was spoken, and we need to find ourselves an appropriate pre-trained model for this task! You’ll remember from the section on pre-trained models for audio classification that Speech Commands is a dataset of spoken words designed to evaluate audio classification models on 15+ simple command words like "up", "down", "yes" and "no", as well as a "silence" label to classify no speech. 


We can take an audio classification model pre-trained on the Speech Commands dataset and pick one of these simple command words to be our chosen wake word. Out of the 15+ possible command words, if the model predicts our chosen wake word with the highest probability, we can be fairly certain that the wake word has been said.

We’ll use  Audio Spectrogram Transformer checkpoint again for our wake word detection task.

```python
from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

#  check what labels the model was trained on
classifier.model.config.id2label

# there is one name in these class labels: id 27 corresponds to the label “marvin”
classifier.model.config.id2label[27]
```

Now we need to define a function that is constantly listening to our device’s microphone input, and continuously passes the audio to the classification model for inference. To do this, we’ll use a handy helper function that comes with 🤗 Transformers called ffmpeg_microphone_live.

This function forwards small chunks of audio of specified length chunk_length_s to the model to be classified. To ensure that we get smooth boundaries across chunks of audio, we run a sliding window across our audio with stride chunk_length_s / 6. So that we don’t have to wait for the entire first chunk to be recorded before we start inferring, we also define a minimal temporary audio input length stream_chunk_s that is forwarded to the model before chunk_length_s time is reached.

The function ffmpeg_microphone_live returns a generator object, yielding a sequence of audio chunks that can each be passed to the classification model to make a prediction. We can pass this generator directly to the pipeline, which in turn returns a sequence of output predictions, one for each chunk of audio input. We can inspect the class label probabilities for each audio chunk, and stop our wake word detection loop when we detect that the wake word has been spoken.


 We’ll use a very simple criteria for classifying whether our wake word was spoken: if the class label with the highest probability was our wake word, and this probability exceeds a threshold prob_threshold, we declare that the wake word as having been spoken. Using a probability threshold to gate our classifier this way ensures that the wake word is not erroneously predicted if the audio input is noise, which is typically when the model is very uncertain and all the class label probabilities low. You might want to tune this probability threshold, or explore more sophisticated means for the wake word decision through an entropy (or uncertainty) based metric.

 
 ```python
from transformers.pipelines.audio_utils import ffmpeg_microphone_live


def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True
# give this function a try to see how it works! We’ll set the flag debug=True to print out the prediction for each chunk of audio.
launch_fn(debug=True)
```

Awesome! As we expect, the model generates garbage predictions for the first few seconds. There is no speech input, so the model makes close to random predictions, but with very low probability. As soon as we say the wake word, the model predicts "marvin" with probability close to 1 and terminates the loop, signalling that the wake word has been detected and that the ASR system should be activated!


### Speech transcription
We’ll use the Whisper Base English for our speech transcription system.

We’ll use a trick to get near real-time transcription by being clever with how we forward our audio inputs to the model. 

```python
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)
```

We can now define a function to record our microphone input and transcribe the corresponding text. With the ffmpeg_microphone_live helper function, we can control how ‘real-time’ our speech recognition model is. Using a smaller stream_chunk_s lends itself to more real-time speech recognition, since we divide our input audio into smaller chunks and transcribe them on the fly. However, this comes at the expense of poorer accuracy, since there’s less context for the model to infer from.

As we’re transcribing the speech, we also need to have an idea of when the user stops speaking, so that we can terminate the recording. For simplicity, we’ll terminate our microphone recording after the first chunk_length_s (which is set to 5 seconds by default), but you can experiment with using a voice activity detection (VAD) model to predict when the user has stopped speaking.

```python
import sys


def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break

    return item["text"]
# Once the microphone is live, start speaking and watch your transcription appear in semi real-time
transcribe()
```

```
Start speaking...
 Hey, this is a test with the whisper model.
```

Nice! You can adjust the maximum audio length chunk_length_s based on how fast or slow you speak (increase it if you felt like you didn’t have enough time to speak, decrease it if you were left waiting at the end), and the stream_chunk_s for the real-time factor. Just pass these as arguments to the transcribe function.

### Language model query
Now that we have our spoken query transcribed, we want to generate a meaningful response. To do this, we’ll use an LLM hosted on the Cloud. Specifically, we’ll pick an LLM on the Hugging Face Hub and use the Inference API to easily query the model.

We’ll use the tiiuae/falcon-7b-instruct checkpoint by TII, a 7B parameter decoder-only LM fine-tuned on a mixture of chat and instruction datasets. You can use any LLM on the Hugging Face Hub that has the “Hosted inference API” enabled, just look out for the widget on the right-side of the model card:


The Inference API allows us to send a HTTP request from our local machine to the LLM hosted on the Hub, and returns the response as a json file. All we need to provide is our Hugging Face Hub token (which we retrieve directly from our Hugging Face Hub folder) and the model id of the LLM we wish to query:

```python
from huggingface_hub import HfFolder
import requests


def query(text, model_id="tiiuae/falcon-7b-instruct"):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HfFolder().get_token()}"}
    payload = {"inputs": text}

    print(f"Querying...: {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"][len(text) + 1 :]

query("What does Hugging Face do?")
```
You’ll notice just how fast inference is using the Inference API - we only have to send a small number of text tokens from our local machine to the hosted model, so the communication cost is very low. The LLM is hosted on GPU accelerators, so inference runs very quickly. Finally, the generated response is transferred back from the model to our local machine, again with low communication overhead.


### Synthesise speech
And now we’re ready to get the final spoken output! Once again, we’ll use the Microsoft SpeechT5 TTS model for English TTS, but you can use any TTS model of your choice. Let’s go ahead and load the processor and model:

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# And also the speaker embeddings
from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# We’ll re-use the synthesise function that we already defined
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

# Let’s quickly verify this works as expected:
from IPython.display import Audio

audio = synthesise(
    "Hugging Face is a company that provides natural language processing and machine learning tools for developers."
)

Audio(audio, rate=16000)
```

### Marvin 🤖

Now that we’ve defined a function for each of the four stages of the voice assistant pipeline, all that’s left to do is piece them together to get our end-to-end voice assistant. We’ll simply concatenate the four stages, starting with wake word detection (launch_fn), speech transcription, querying the LLM, and finally speech synthesis.

```python
launch_fn()
transcription = transcribe()
response = query(transcription)
audio = synthesise(response)

Audio(audio, rate=16000, autoplay=True)
```

And with that, we have our end-to-end voice assistant complete, made using the 🤗 audio tools you’ve learnt throughout this course, with a sprinkling of LLM magic at the end. There are several extensions that we could make to improve the voice assistant. Firstly, the audio classification model classifies 35 different labels. We could use a smaller, more lightweight binary classification model that only predicts whether the wake word was spoken or not. Secondly, we pre-load all the models ahead and keep them running on our device. If we wanted to save power, we would only load each model at the time it was required, and subsequently un-load them afterwards. Thirdly, we’re missing a voice activity detection model in our transcription function, transcribing for a fixed amount of time, which in some cases is too long, and in others too short.


### Generalise to anything 🪄

So far, we’ve seen how we can generate speech outputs with our voice assistant Marvin. To finish, we’ll demonstrate how we can generalise these speech outputs to text, audio and image.

We’ll use Transformers Agents to build our assistant. Transformers Agents provides a natural language API on top of the 🤗 Transformers and Diffusers libraries, interpreting a natural language input using an LLM with carefully crafted prompts, and using a set of curated tools to provide multimodal outputs.

Let’s go ahead and instantiate an agent. There are three LLMs available for Transformers Agents, two of which are open-source and free on the Hugging Face Hub. The third is a model from OpenAI that requires an OpenAI API key. We’ll use the free Bigcode Starcoder model in this example, but you can also try either of the other LLMs available:

```python
from transformers import HfAgent

agent = HfAgent(
    url_endpoint="https://api-inference.huggingface.co/models/bigcode/starcoder"
)
```
To use the agent, we simply have to call agent.run with our text prompt. As an example, we’ll get it to generate an image of a cat 🐈 (that hopefully looks a bit better than this emoji):

```python
agent.run("Generate an image of a cat")
```

Easy as that! The Agent interpreted our prompt, and used Stable Diffusion under the hood to generate the image, without us having to worry about loading the model, writing the function or executing the code.

We can now replace our LLM query function and text synthesis step with our Transformers Agent in our voice assistant, since the Agent is going to take care of both of these steps for us:

```python
launch_fn()
transcription = transcribe()
agent.run(transcription)
```

Try speaking the same prompt “Generate an image of a cat” and see how the system gets on. If you ask the Agent a simple question / answer query, the Agent will respond with a text answer. You can encourage it to generate multimodal outputs by asking it to return an image or speech. For example, you can ask it to: “Generate an image of a cat, caption it, and speak the caption”.

While the Agent is more flexible than our first iteration Marvin 🤖 assistant, generalising the voice assistant task in this way may lead to inferior performance on standard voice assistant queries. To recover performance, you can try using a more performant LLM checkpoint, such as the one from OpenAI, or define a set of custom tools that are specific to the voice assistant task.



# III. Transcribe a meeting

In this final section, we’ll use the Whisper model to generate a transcription for a conversation or meeting between two or more speakers. We’ll then pair it with a speaker diarization model to predict “who spoke when”. By matching the timestamps from the Whisper transcriptions with the timestamps from the speaker diarization model, we can predict an end-to-end meeting transcription with fully formatted start / end times for each speaker. This is a basic version of the meeting transcription services you might have seen online from the likes of Otter.ai and co:


### Speaker Diarization

Speaker diarization (or diarisation) is the task of taking an unlabelled audio input and predicting “who spoke when”. In doing so, we can predict start / end timestamps for each speaker turn, corresponding to when each speaker starts speaking and when they finish.

🤗 Transformers currently does not have a model for speaker diarization included in the library, but there are checkpoints on the Hub that can be used with relative ease. 

In this example, we’ll use the pre-trained speaker diarization model from pyannote.audio. Let’s get started and pip install the package:

```python
pip install --upgrade pyannote.audio
```
Great! The weights for this model are hosted on the Hugging Face Hub. To access them, we first have to agree to the speaker diarization model’s terms of use: pyannote/speaker-diarization. And subsequently the segmentation model’s terms of use: pyannote/segmentation.

Once complete, we can load the pre-trained speaker diarization pipeline locally on our device:
```python
from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=True
)
```

Let’s try it out on a sample audio file! For this, we’ll load a sample of the LibriSpeech ASR dataset that consists of two different speakers that have been concatenated together to give a single audio file:

```python
from datasets import load_dataset

concatenated_librispeech = load_dataset(
    "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
)
sample = next(iter(concatenated_librispeech))
```
We can listen to the audio to see what it sounds like:

```python
from IPython.display import Audio

Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

Cool! We can clearly hear two different speakers, with a transition roughly 15s of the way through. Let’s pass this audio file to the diarization model to get the speaker start / end times. Note that pyannote.audio expects the audio input to be a PyTorch tensor of shape (channels, seq_len), so we need to perform this conversion prior to running the model:

```python
import torch

input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()
outputs = diarization_pipeline(
    {"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]}
)

outputs.for_json()["content"]
```

```
[{'segment': {'start': 0.4978125, 'end': 14.520937500000002},
  'track': 'B',
  'label': 'SPEAKER_01'},
 {'segment': {'start': 15.364687500000002, 'end': 21.3721875},
  'track': 'A',
  'label': 'SPEAKER_00'}]
```
This looks pretty good! We can see that the first speaker is predicted as speaking up until the 14.5 second mark, and the second speaker from 15.4s onwards. Now we need to get our transcription!


### Speech transcription
For the third time in this Unit, we’ll use the Whisper model for our speech transcription system. Specifically, we’ll load the Whisper Base checkpoint, since it’s small enough to give good inference speed with reasonable transcription accuracy. As before, feel free to use any speech recognition checkpoint on the Hub, including Wav2Vec2, MMS ASR or other Whisper checkpoints:

```python
from transformers import pipeline

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)
```
Let’s get the transcription for our sample audio, returning the segment level timestamps as well so that we know the start / end times for each segment. You’ll remember from Unit 5 that we need to pass the argument return_timestamps=True to activate the timestamp prediction task for Whisper:

```python
asr_pipeline(
    sample["audio"].copy(),
    generate_kwargs={"max_new_tokens": 256},
    return_timestamps=True,
)
```

```
{
    "text": " The second and importance is as follows. Sovereignty may be defined to be the right of making laws. In France, the king really exercises a portion of the sovereign power, since the laws have no weight. He was in a favored state of mind, owing to the blight his wife's action threatened to cast upon his entire future.",
    "chunks": [
        {"timestamp": (0.0, 3.56), "text": " The second and importance is as follows."},
        {
            "timestamp": (3.56, 7.84),
            "text": " Sovereignty may be defined to be the right of making laws.",
        },
        {
            "timestamp": (7.84, 13.88),
            "text": " In France, the king really exercises a portion of the sovereign power, since the laws have",
        },
        {"timestamp": (13.88, 15.48), "text": " no weight."},
        {
            "timestamp": (15.48, 19.44),
            "text": " He was in a favored state of mind, owing to the blight his wife's action threatened to",
        },
        {"timestamp": (19.44, 21.28), "text": " cast upon his entire future."},
    ],
}
```

Alright! We see that each segment of the transcript has a start and end time, with the speakers changing at the 15.48 second mark. We can now pair this transcription with the speaker timestamps that we got from our diarization model to get our final transcription.

### Speechbox

To get the final transcription, we’ll align the timestamps from the diarization model with those from the Whisper model. The diarization model predicted the first speaker to end at 14.5 seconds, and the second speaker to start at 15.4s, whereas Whisper predicted segment boundaries at 13.88, 15.48 and 19.44 seconds respectively. Since the timestamps from Whisper don’t match perfectly with those from the diarization model, we need to find which of these boundaries are closest to 14.5 and 15.4 seconds, and segment the transcription by speakers accordingly. Specifically, we’ll find the closest alignment between diarization and transcription timestamps by minimising the absolute distance between both.

Luckily for us, we can use the 🤗 Speechbox package to perform this alignment. First, let’s pip install speechbox from main:

```
pip install git+https://github.com/huggingface/speechbox
```
We can now instantiate our combined diarization plus transcription pipeline, by passing the diarization model and ASR model to the ASRDiarizationPipeline class:

```
from speechbox import ASRDiarizationPipeline

pipeline = ASRDiarizationPipeline(
    asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline
)

# pass the audio file to the composite pipeline and see what we get out:
pipeline(sample["audio"].copy())
```

```
[{'speaker': 'SPEAKER_01',
  'text': ' The second and importance is as follows. Sovereignty may be defined to be the right of making laws. In France, the king really exercises a portion of the sovereign power, since the laws have no weight.',
  'timestamp': (0.0, 15.48)},
 {'speaker': 'SPEAKER_00',
  'text': " He was in a favored state of mind, owing to the blight his wife's action threatened to cast upon his entire future.",
  'timestamp': (15.48, 21.28)}]
```
Excellent! The first speaker is segmented as speaking from 0 to 15.48 seconds, and the second speaker from 15.48 to 21.28 seconds, with the corresponding transcriptions for each.

We can format the timestamps a little more nicely by defining two helper functions. The first converts a tuple of timestamps to a string, rounded to a set number of decimal places. The second combines the speaker id, timestamp and text information onto one line, and splits each speaker onto their own line for ease of reading:

```
def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(raw_segments):
    return "\n\n".join(
        [
            chunk["speaker"] + " " + tuple_to_string(chunk["timestamp"]) + chunk["text"]
            for chunk in raw_segments
        ]
    )
```

Let’s re-run the pipeline, this time formatting the transcription according to the function we’ve just defined:

```
outputs = pipeline(sample["audio"].copy())

format_as_transcription(outputs)
```

```
SPEAKER_01 (0.0, 15.5) The second and importance is as follows. Sovereignty may be defined to be the right of making laws.
In France, the king really exercises a portion of the sovereign power, since the laws have no weight.

SPEAKER_00 (15.5, 21.3) He was in a favored state of mind, owing to the blight his wife's action threatened to cast upon
his entire future.
```

There we go! With that, we’ve both diarized and transcribe our input audio and returned speaker-segmented transcriptions. While the minimum distance algoirthm to align the diarized timestamps and transcribed timestamps is simple, it works well in practice. If you want to explore more advanced methods for combining the timestamps, the source code for the ASRDiarizationPipeline is a good place to start: speechbox/diarize.py.






