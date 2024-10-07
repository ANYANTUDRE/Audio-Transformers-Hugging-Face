# I. Speech-to-speech translation

**Speech-to-speech translation (STST or S2ST)** is a relatively new spoken language processing task involving translating speech from one langauge into speech in a different language.

![]()

STST holds applications in the field of multilingual communication.
2 possible approaches:
- **Two stage cascaded approach to STST**: speech translation (ST) + text-to-speech (TTS).
  Advantages:
    - straightforward, it results in very effective STST systems.
    - very data and compute efficient

![]()

- **Three stage approach**: ASR + MT + TTS.
Problems:
  - error propagation through the pipeline
  - increase in latency

### Speech translation

We’ll use the [Whisper Base - 74M]() model for our speech translation system, since it’s capable of translating from over 96 languages to English.
Advantage: get reasonable inference speed

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

We’ll use the pre-trained SpeechT5 TTS model for English TTS.
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

Let’s do a quick sanity check to make sure we can concatenate the two models, putting an audio sample in and getting an audio sample out.

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







