# I. Speech-to-speech translation

Speech-to-speech translation (STST or S2ST) is a relatively new spoken language processing task. 
It involves translating speech from one langauge into speech in a different language:

![]()

STST holds applications in the field of multilingual communication, enabling speakers in different languages to communicate with one another through the medium of speech.
(more natural way of communicating compared to text-based machine translation).

We’ll explore a cascaded approach to STST: use a speech translation (ST) system to transcribe the source speech into text in the target language, then text-to-speech (TTS) to generate speech in the target language from the translated text:

![]()


We could also have used a three stage approach, where first we use an ASR system to transcribe the source speech into text in the same language, 
then MT to translate the transcribed text into the target language, and finally TTS to generate speech in the target language.
Problems with this approach:
- error propagation through the pipeline
- increase in latency







