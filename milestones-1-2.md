---
description: Build and test Milestone 1 (baselines, measurement rig, actuators) and Milestone 2 (wake word QAT, distillation, exit heads) for the offline voice assistant
argument-hint: "[m1 | m2 | all]  (default: all)"
---

You are working on the offline early exit voice assistant described in CLAUDE.md. Read CLAUDE.md
first and follow every rule in it, especially: the novel parts must be written from scratch,
deployment is TFLite only, nothing leaves the device, and never fake or hide a result.

Scope for this run: **$ARGUMENTS** (if empty, do M1 then M2).

## How to work
1. Read CLAUDE.md, `configs/device.yaml` and the current repo. Make a short plan and a task list
   with one task per numbered item below. If `configs/device.yaml` has no `target`, stop and ask.
2. Work through the items in order. After each item, run its tests. Do not move on while its
   tests fail. If something is blocked (no GPU, no device, dataset not downloaded, hardware not
   wired), write the blocker in `docs/milestones.md`, skip what depends on it, and keep going
   with the rest.
3. Commit after each finished item with a clear message.
4. At the end, write `docs/milestones.md` with a status table (item, status, test result,
   numbers, blockers) and append to `docs/AI_USE.md` what you wrote.

Ask me before: installing system packages on the device, downloading anything over 5 GB,
changing the pipeline design, or deleting files.

---

## Milestone 1: baselines on device, measurement rig, actuators

**1.1 Repo skeleton and configs**
- Create the layout from CLAUDE.md, `pyproject.toml` or `requirements.txt` (separate
  `requirements-train.txt` for GPU and `requirements-device.txt` for the device), `pytest.ini`
  with the `device` and `gpu` markers, and `.gitignore` for data and checkpoints.
- Configs: `device.yaml`, `train_wakeword.yaml`, `train_encoder.yaml`, `export.yaml`, `eval.yaml`.
- Tests: `tests/test_configs.py` loads every config and checks required keys.

**1.2 Data preparation**
- Scripts in `data/` to download and prepare Speech Commands v2, FSC, and the SLURP `iot` subset.
- Shared feature extraction in `src/audio/features.py`: 16 kHz mono, 80 bin log mel (Whisper
  compatible), fixed length padding or cropping to the configured window (4 to 5 s for the encoder,
  1 s for the wake word).
- Label maps for FSC (action, object, location) saved as JSON.
- Tests: `tests/test_data.py`: feature shape and dtype on a synthetic sine wave; padding and
  cropping lengths; label map round trip; a sample count check on each prepared split
  (skip with a clear message if the dataset is not downloaded).

**1.3 Live microphone capture**
- `src/audio/mic.py`: streaming capture with a ring buffer (about 2 s), configurable device index,
  plus a simple energy based VAD.
- Tests: `tests/test_mic.py`: ring buffer overwrite logic on synthetic frames (no hardware);
  a `device` marked test that records 1 s and checks the RMS is above a noise floor.

**1.4 Naive baseline (FP32)**
- `src/baselines/naive.py`: Whisper tiny encoder, FP32, full 4 layers, mean pooled features plus a
  linear intent classifier trained on FSC. This is the "no tricks" reference.
- Export to FP32 TFLite (`export/naive_fp32.tflite`).
- Tests: `tests/test_baseline.py`: model forward shape on random input; the TFLite model output
  matches PyTorch within tolerance (max abs diff < 1e-3) on 5 random inputs; a `gpu` test that
  checks training loss decreases over 50 steps.

**1.5 Measurement harness**
- `src/measure/latency.py`: warmup runs, then N timed runs; report mean, p50, p90, p99 in ms.
  Pin threads from config. Log CPU governor and temperature on the Pi.
- `src/measure/memory.py`: peak RSS during inference.
- `src/measure/energy.py`: interface for the INA226 logger plus a GPIO marker that goes high
  during inference. If no INA226 is present, fall back to a stub that records "not measured",
  never a made up value. On Android use BatteryManager readings.
- Every script writes CSV to `results/<run_name>/` with the git hash and device info.
- Tests: `tests/test_measure.py`: latency harness on a dummy sleep function returns plausible
  numbers; CSV has the expected columns; the energy stub reports "not measured";
  `device` tests that run the naive baseline on the device and check that latency and RSS are
  recorded and nonzero.

**1.6 Actuators and device registry**
- `src/actuators/registry.py`: the list of devices, allowed actions and value ranges
  (lights on/off/brightness, curtains open/close, thermostat setpoint 16 to 30 C).
- `src/actuators/client.py`: MQTT client (or BLE, from config) that sends validated commands.
- `src/actuators/mock.py`: an in memory mock broker for tests.
- If the target is a Pi, also `src/actuators/gpio.py` to drive the LED, servo and BME280 directly.
- Tests: `tests/test_actuators.py`: valid commands pass; unknown device, bad action and out of range
  values are rejected; the mock receives the exact payload; a `device` test that toggles the light
  and reads the BME280.

**M1 done when:** `pytest -m "not device and not gpu"` passes, the naive baseline runs on the device
with latency and memory logged to CSV, and a light toggles from a test command.

---

## Milestone 2: wake word QAT, distillation, exit heads

**2.1 Wake word model with QAT**
- `src/wakeword/model.py`: DS CNN (small, under 100k parameters to start; widths from config).
- Train on Speech Commands v2 with one chosen wake word versus everything else plus background noise.
- QAT to int8 (TF Model Optimization Toolkit, or PyTorch QAT then convert). Export full integer int8 TFLite.
- Report accuracy, false reject rate, and false accepts per hour on the background noise set,
  for FP32 vs int8.
- Tests: `tests/test_wakeword.py`: forward shape; parameter count under the limit; the exported
  model is really int8 (check input and output dtypes and tensor types in the interpreter);
  FP32 vs int8 top 1 agreement above 95% on a held out batch; a `gpu` test that loss decreases.

**2.2 Student encoder and distillation**
- `src/encoder/student.py`: Whisper tiny encoder only, input window shortened to the configured
  length, positional embeddings cropped to match.
- `src/encoder/distill.py`: Whisper base as teacher (fine tuned on FSC first). Loss = task loss
  plus KL on logits (temperature from config) plus feature matching on projected hidden states.
- Train a "no distillation" version with the same seed and budget for the later ablation.
- Tests: `tests/test_distill.py`: shortened input gives the right sequence length; KL loss is zero
  when teacher and student logits match and positive otherwise; feature projection shapes line up;
  a `gpu` test that distillation loss decreases over 100 steps.

**2.3 Exit heads, decision head and asymmetric loss (novel, write from scratch)**
- `src/encoder/exits.py`: heads after layers 2, 3 and 4. Each has an intent head, a slot or value
  head, and a decision head over {ACT, CONTINUE, REJECT}. The last exit has no CONTINUE.
- `src/encoder/loss.py`: the asymmetric action loss. A cost matrix where acting on the wrong
  device or action costs lambda, rejecting costs 1, and acting correctly costs 0. Weighted sum
  over exits. Also distil the final exit into the earlier ones.
- `src/encoder/policy.py`: the inference time policy that walks the exits and stops on ACT or REJECT.
- Train with lambda in {1, 2, 5, 10} and save each checkpoint (this is the main ablation later).
- Tests: `tests/test_exits.py` and `tests/test_loss.py`:
  - each head outputs the right shape at each exit
  - with lambda = 1 the loss reduces to symmetric weighting; with lambda > 1 a wrong ACT is
    penalized exactly lambda times a REJECT on a hand built example
  - the loss is differentiable (gradients flow to every head)
  - the policy stops at the first ACT, rejects correctly, never returns CONTINUE at the last exit,
    and returns which exit it used
  - a `gpu` test that higher lambda gives a lower false action rate on the validation split
    (this is the hypothesis; if it fails, record it, do not tune it away)

**2.4 Segmented TFLite export (novel, write from scratch)**
- `src/export/segments.py`: export the encoder as separate int8 TFLite models: frontend plus
  layers 1 to 2 with exit 1 heads, layer 3 with exit 2 heads, layer 4 with exit 3 heads.
  Hidden states pass between segments in Python.
- Representative dataset calibration from FSC train audio.
- `src/runtime/pipeline.py`: wake word, then segments, then policy, then registry check, then actuator.
- Tests: `tests/test_export.py`:
  - chaining all segments gives the same intent as the full PyTorch model on at least 95% of a
    held out batch
  - when the policy exits at layer 2, the later segment interpreters are never invoked
    (count invocations)
  - all segment files are int8, and their total size is logged
  - a `device` test that measures latency per exit on the device and checks exit at layer 2 is
    faster than running all layers

**M2 done when:** all non device tests pass, int8 wake word and segmented encoder files exist
in `export/`, a results table compares naive FP32 vs distilled int8 early exit (accuracy, false
action rate, latency per exit, peak memory) for each lambda, and the device tests pass or their
blockers are written down.

---

## Final report for this run
In `docs/milestones.md`, give:
- a status table for every numbered item
- the key numbers (on device latency, memory, accuracy, false action rate) with the CSV path for each
- what failed or was skipped and why
- what I need to do by hand (wiring, downloads, device setup)
- what to show at the mid project presentation
