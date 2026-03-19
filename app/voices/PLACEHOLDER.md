# Voice Preset Placeholders

The WAV files in this directory are **silent placeholders**.

They are valid 1-second 24 kHz mono WAV files, but contain no speech.
You **must** replace them with real voice samples before running the server in
production, or voice cloning will produce silence.

## How to replace

1. Record or source a clean speech sample for each voice character:
   - `Alice.wav` — Female, neutral American English (maps to OpenAI `alloy`)
   - `Echo.wav` — Male, neutral American English (maps to OpenAI `echo`)
   - `Frank.wav` — Male, British English (maps to OpenAI `fable`)
   - `Onyx.wav` — Male, deep American English (maps to OpenAI `onyx`)
   - `Nova.wav` — Female, warm American English (maps to OpenAI `nova`)
   - `Shimmer.wav` — Female, soft American English (maps to OpenAI `shimmer`)

2. Requirements per sample:
   - Format: WAV (PCM), 24 kHz preferred (other rates are accepted)
   - Duration: 5–30 seconds
   - Clean speech — no background music, minimal noise
   - Mono or stereo

3. Replace the file(s) and rebuild the Docker image.

## Important

Only use voice samples for which you have the legal right to use the speaker's
voice.  Do not use recordings of real people without their explicit, recorded
consent.  See `docs/README.md#responsible-use` for the full policy.
