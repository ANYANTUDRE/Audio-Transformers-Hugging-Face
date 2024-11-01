### Unit 2. A gentle introduction to audio applications

Previously, we explored the fundamentals of audio data and learned how to work with audio datasets using the 🤗 Datasets and 🤗 Transformers libraries.  

At this point you may be eager to learn about the audio tasks that 🤗 Transformers can handle, and you have all the foundational knowledge necessary to dive in! Let’s take a look at some of the mind-blowing audio task examples:

- **Audio classification:** easily categorize audio clips into different categories. You can identify whether a recording is of a barking dog or a meowing cat, or what music genre a song belongs to.
- **Automatic speech recognition:** transform audio clips into text by transcribing them automatically. You can get a text representation of a recording of someone speaking, like “How are you doing today?“. Rather useful for note taking!
- **Speaker diarization:** Ever wondered who’s speaking in a recording? With 🤗 Transformers, you can identify which speaker is talking at any given time in an audio clip. Imagine being able to differentiate between “Alice” and “Bob” in a recording of them having a conversation.
- **Text to speech** create a narrated version of a text that can be used to produce an audio book, help with accessibility, or give a voice to an NPC in a game. With 🤗 Transformers, you can easily do that!


In this unit, you’ll learn how to use pre-trained models for some of these tasks using the pipeline() function from 🤗 Transformers. Specifically, we’ll see how the pre-trained models can be used for audio classification, automatic speech recognition and audio generation. Let’s get started!
