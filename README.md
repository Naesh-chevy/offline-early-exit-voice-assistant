# Offline early exit voice assistant

A voice assistant for smart home lights that runs fully offline. It listens on a live microphone,
understands commands like "turn on the kitchen lights", and switches a light on a Raspberry Pi.
No audio, transcript or command leaves the device. There is no cloud path and no LLM in the pipeline.

This is a course project for ECE 690OD (On Device Deep Learning) at UMass Amherst.

## Scope

- Commands: lights on and off in the kitchen, bedroom and living room.
- FSC has no "living room", so a command with no room (for example "turn on the lights") means the living room.
- Target device: Raspberry Pi 4B (64 bit Raspberry Pi OS). The setting is in `configs/device.yaml`.
- The first plan also had curtains and a thermostat. We dropped them, and some old notes in `docs/` still mention them.

## How it works

1. **Wake word.** A small depthwise separable CNN listens for "marvin". It is trained with quantization aware
   training and exported as an int8 TFLite model (33 KB).
2. **Encoder.** A Whisper tiny encoder (4 layers) reads about 4 seconds of audio instead of 30. It was distilled
   from a Whisper base teacher that we fine tuned on Fluent Speech Commands. There is no decoder.
3. **Early exits.** There are exits after encoder layers 2, 3 and 4. Each exit has an intent head (action, object,
   location) and a decision head with three outcomes: ACT (confident, stop), CONTINUE (run the next layer) and
   REJECT (ask the user to repeat).
4. **Segmented runtime.** The encoder is exported as three separate int8 TFLite segments. When an exit says ACT or
   REJECT, the later segments never run.
5. **Validation.** The predicted command is checked against a device registry before anything is sent.
6. **Actuators.** The command goes out over MQTT. A listener on the Pi drives an LED strip through a MOSFET on GPIO.

The parts we wrote ourselves (no early exit library): the exit heads, the ACT / CONTINUE / REJECT decision logic,
the asymmetric action loss, the threshold policy and the segmented TFLite runtime.

The asymmetric loss charges a wrong device action lambda times more than a reject. We tried lambda = 1, 2, 5 and 10.

## Results so far

Test set of Fluent Speech Commands, 3793 clips. Thresholds were picked on the validation split. One training seed.

| System | Wrong action | Reject | Mean exit (1 = layer 2, 3 = layer 4) |
|---|---|---|---|
| Naive baseline (fp32 Whisper tiny, no exits) | 1.24% | none | n/a |
| Full encoder + reject, lambda 10 | 0.90% | 0.74% | 3.00 |
| Early exit + reject, lambda 10 | 0.92% | 0.66% | 2.74 |
| Early exit + reject, lambda 5 | 1.27% | 1.13% | 2.58 |
| Early exit, lambda 2 | 2.14% | 0.08% | 1.07 |
| Early exit, lambda 1 | 2.53% | 0.03% | 1.00 |

Other numbers:

- int8 costs 0 to 0.1 points of accuracy. The encoder segments are 8.3 MB in int8 and 31.4 MB in fp32.
- Wake word (int8): 99.35% accuracy, 7.2% false rejects, about 90 false accepts per hour. That last number comes
  from only about 80 seconds of held out noise, so treat it as a rough upper bound.
- Early exit saves about 6 to 10% of the encoder layers. That is a small saving, and it costs no accuracy at lambda 10.
- The difference between early exit and the naive baseline is about 12 clips out of 3793 on one seed. We do not
  claim it is statistically significant yet.

Not done yet: latency, memory and energy on the Pi, a second benchmark (SLURP), multiple seeds, and the
QUADS style prior method comparison (the code is written in `src/baselines/quads.py` but not trained).

## Repo layout

```
configs/            yaml configs (device, training, export, eval)
data/               scripts to download and prepare the datasets (data itself is not in git)
src/audio/          mic capture, VAD, log mel features
src/wakeword/       DS CNN model, quantization aware training, export
src/encoder/        Whisper tiny student, distillation, exit heads, asymmetric loss, policy, evaluation
src/export/         segmented TFLite export helpers
src/runtime/        on device pipeline and the segmented runtime
src/actuators/      device registry, MQTT client, GPIO, mock actuators
src/measure/        latency, memory and energy tools
src/baselines/      naive fp32 baseline and the QUADS style prior method
tests/              pytest suite
results/            CSV logs, one folder per run
docs/               notes, AI use disclosure, milestone reports
gpu_package/        Colab notebook and instructions for training on a GPU
```

## Setup and tests

Python 3.11 is the target. Install the laptop requirements and run the tests that need no GPU or Pi:

```bash
pip install -r requirements.txt
python -m pytest -m "not device and not gpu"
```

Tests marked `device` need the real Raspberry Pi, and tests marked `gpu` need a GPU.

## Training

Datasets and trained models are not stored in this repo. Training was done on a Colab GPU. Follow
`docs/GPU_TRAINING.md` or use `gpu_package/colab_training.ipynb`. The main steps are:

```bash
python data/prepare.py sc
python data/prepare.py fsc
python -m src.wakeword.train --run_name wakeword
python -m src.encoder.distill --stage teacher
python -m src.encoder.distill --stage student
python -m src.encoder.train_exits --encoder checkpoints/student/encoder.pt --tag distill
python -m src.baselines.naive
python -m src.encoder.eval_exits --lams 1 2 5 10 --int8 5 10
```

## Datasets

- Google Speech Commands v2 for the wake word.
- Fluent Speech Commands for the intents (action, object, location).

Both have their own terms of use. Download them yourself with the scripts in `data/`.

## AI use

AI was used in this project. Claude (Claude Sonnet 5.5) helped write the code, tests and docs, see `docs/AI_USE.md`.
Short notes on how each module works are in `docs/notes.md`.

## Licence

MIT, see `LICENSE`.
