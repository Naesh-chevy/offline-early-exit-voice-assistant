# Milestones 1 and 2: status report (run on 2026-11-03)

> Scope update: the project is now lights on and off in three rooms only. Curtains, the thermostat and the numeric setpoint are dropped, so mentions of them below are from the first plan and are not needed.

Target device: Raspberry Pi 4B (`configs/device.yaml`, `target: pi4`, 4 threads).
All tests were run on the Windows laptop (CPU only, Python 3.13, no GPU, no Pi). Nothing has been run on the Pi.

**Short version: all code and tests for items 1.1 to 2.4 are written and the laptop test suite passes
(`pytest -m "not device and not gpu"`: 69 passed, 0 skipped, 8 deselected after all three downloads). Neither milestone is done by the
definitions in the task file, because no dataset has been downloaded, nothing has been trained, and nothing has run on
the Pi. There are no real accuracy, latency, memory or energy numbers yet.**

## Status table

| Item | What | Status | Test result (laptop) | Numbers | Blocker |
|---|---|---|---|---|---|
| 1.1 | Skeleton, configs | done | `test_configs` pass | none | none |
| 1.2 | Data scripts, log mel, label maps | code done; Speech Commands v2, FSC and SLURP (real audio, iot only) downloaded and prepared | all 8 data tests pass, none skipped | SC v2: train 84843 clips (1710 wake word "marvin"), valid 9981 (195), test 11005 (195). FSC: 23132 / 3118 / 3793 clips (matches the published sizes). SLURP iot: 3676 train / 483 valid / 1093 test recordings, 0 missing audio files | slurp_synth not downloaded (not needed) |
| 1.3 | Mic, ring buffer, VAD | code done | ring buffer and VAD tests pass; device mic test not run | none | needs Pi with mic |
| 1.4 | Naive FP32 baseline | code done, not trained | forward shape, TFLite vs PyTorch (max abs diff < 1e-3, 5 inputs), CPU loss decrease pass; gpu test not run | none | needs FSC + GPU to train |
| 1.5 | Measurement harness | code done | latency, RSS, CSV (has git_hash), energy "not measured" tests pass; device test not run | none | needs Pi; INA226 not wired |
| 1.6 | Registry, MQTT client, mock, GPIO | code done | valid, rejected, exact payload tests pass; device light/BME280 test not run | none | needs Pi wiring |
| 2.1 | Wake word DS-CNN + QAT + int8 | code done, not trained | shape, params < 100k, really int8 (in/out dtype int8, no float32 tensors), fp32 vs int8 agreement > 95% on synthetic batch, CPU loss decrease pass; gpu test not run | none on real data | needs Speech Commands v2 + GPU |
| 2.2 | Student, distillation | code done, not trained | sequence length 200 for 4 s, KL zero when equal and positive otherwise, projection shapes, CPU distill loss decrease pass; gpu test not run | none | needs FSC + GPU, internet for pretrained weights |
| 2.3 | Exits, asymmetric loss, policy | code done, not trained | all shape, loss and policy tests pass; gpu lambda test not run | toy only, see below | needs FSC + GPU for the real lambda sweep |
| 2.4 | Segmented int8 export, pipeline | code done, not exported from a trained model | chain vs PyTorch, early exit never calls later segments, int8 files, pipeline-with-mock tests pass; device latency test not run | sizes of random weights, see below | needs trained checkpoint, Pi |

## Numbers that exist (and what they are not)

These are from synthetic data or random weights. They only show the code works, they say nothing about the project's accuracy.

- Segmented int8 export of the real size student (d=384, 4 layers, 200 positions), random weights, calibrated on 8 random clips:
  seg1 4386 KB, seg2 1942 KB, seg3 1942 KB, total 8271 KB. Export took about 7 s on the laptop. Conversion works at full size.
- Tiny synthetic model, int8 chain vs PyTorch, same intent on 100 of 100 held out samples. Only shows the plumbing is right.
  Real int8 agreement on the trained model is unknown. Transformer int8 can lose accuracy, so test it for real.
- Toy lambda check on CPU (`test_higher_lambda_lowers_false_action_rate_cpu_tiny`, synthetic data with 30% random action
  labels so mistakes are unavoidable, 150 steps, tiny model): false action rate 0.295 for lambda 1 and 0.0 for lambda 10.
  The lambda 10 model gets there by rejecting everything, which is what the cost table asks for on unlearnable noise. The
  direction matches the hypothesis, but this is a toy and says nothing about the real result. The real test is the gpu one on
  FSC, not run yet. If it fails, report it, do not tune it away. (An earlier version of this test used clean labels and
  gave 0.0 for both, so it proved nothing. I changed the data and made the assert strict.)
- Wake word model size: default widths give well under 100k parameters (test checks this).

No CSV files were written to `results/` because no real measurement was made.

## What failed or was skipped, and why

- **Downloaded**: SLURP annotations (GitHub) and slurp_real audio (3.9 GB, Zenodo record 4274930), prepared with `prepare.py slurp` (iot only). `slurp_synth` (2.8 GB) was not downloaded because `prepare.py` does not use it. FSC: the fluent.ai download link is dead, so we use the Zenodo copy (record 14722453, third party, 5 tars, all MD5s verified). `data/prepare.py fsc` reads the tars directly and gives exactly 23132 / 3118 / 3793 clips, with 6 action, 14 object and 4 location classes. A CPU smoke test ran the pretrained Whisper tiny and base, a few distillation steps and one exit-head step on real FSC clips with no errors (no accuracy claimed). Speech Commands v2 was downloaded on 2026-11-03 and `data/prepare.py sc` ran fine. `WakeDataset` loads real clips (about 7 ms each) and the 6 background noise files split 320 s for training and 80 s held out. 80 s is very little for false accepts per hour: one false hit would read as about 45 per hour, so treat that number as coarse.
- **Not trained**: teacher fine tune, student distillation, no-distillation ablation model, wake word, exit heads for
  lambda 1, 2, 5, 10. No GPU on this laptop.
- **Not run on the Pi**: the 4 `device` tests (mic, naive baseline latency and RSS, light toggle + BME280, latency per
  exit). Also no GPU tests ran.
- **Not written**: BLE transport (raises NotImplementedError), OLED setpoint display, the QUADS style prior method
  comparison (not part of items 1.1 to 2.4), the results table of naive FP32 vs distilled int8 per lambda (needs trained models).
- **FSC has no curtains and no numeric values.** The label maps come from FSC, so the object head has 14 classes and no
  "curtains". The registry and `intent_to_command` already handle curtains and thermostat values, but the model cannot
  predict them until `data/own/` recordings exist and the label maps and head sizes are extended. The slot (setpoint) head
  is only trained on samples with a `value` column (own recordings), fsc samples are ignored for it.
- **Design choices to know about** (also in `docs/notes.md`):
  - Python 3.13 venv, not 3.11 as CLAUDE.md says, because only 3.13 is installed. TensorFlow 2.21 and torch run on it.
  - QAT is our own fake quant in PyTorch, then TFLite does post training quantization on those weights. TFLite chooses its own
    activation scales. `tensorflow-model-optimization` does not work with Keras 3.
  - `ai-edge-torch` is Linux only, so the TFLite graphs are rebuilt by hand in TF ops from the PyTorch weights.
  - TF 2.21 has no `tf.lite.Interpreter`, so everything uses `ai-edge-litert` (same package the Pi gets).
  - Naive baseline uses the same 4 s input window as the real system.
  - The CONTINUE cost (0.2) is a new hyperparameter in `configs/train_encoder.yaml`. Policy thresholds in the same file are
    placeholders until the exit and threshold ablation tunes them on the validation set.
  - No git commits were made (you chose no commits, and this folder is not a git repo). CSVs will say `no-git` in the hash column
    until you run `git init` and commit.
  - `data/prepare.py` has not been run on real files, so the FSC and SLURP csv column names are from memory of the dataset formats. Check on first run.

## What you need to do by hand

1. Download data (Speech Commands is done): get FSC and unzip to `data/raw/fsc/` (needs `data/` and `wavs/`), SLURP `iot` if you want the second benchmark. Then `python data/prepare.py slurp` (`sc` and `fsc` are done).
2. On the GPU box: `pip install -r requirements-train.txt`, then
   `python -m src.encoder.distill --stage teacher`, `--stage student`, `--stage student --no_distill`,
   `python -m src.encoder.train_exits --encoder checkpoints/student/encoder.pt --tag distill` (and the same for `--tag nodistill`),
   `python -m src.wakeword.train`.
3. Export: `python -m src.export.segments --ckpt checkpoints/exits_distill_lam5.pt` and the naive model (`python -m src.baselines.naive` exports `export/naive_fp32.tflite`). Copy `export/` to the Pi.
4. On the Pi: `pip install -r requirements-device.txt`, install and start a Mosquitto broker (system package, ask first), wire LED via MOSFET on GPIO 18, servo on 12, heater LED on 23, marker on 24, BME280 on I2C 0x76, INA226 on the 5 V line if you want energy. Then `pytest -m device`.
5. Run `python -m src.measure.bench --model export/naive_fp32.tflite --run_name naive_fp32` and the same for each segment for the latency, memory and energy CSVs.
6. Record curtain and numeric (setpoint) clips into `data/own/` and extend the label maps.
7. `git init` and commit, if you want history and the git hash in the CSVs.

## What to show at the mid project presentation (Nov 4 / Nov 6)

We are behind the schedule: M1 and M2 were due Oct 9 and Oct 30. Only show what is true:
- The pipeline diagram and the design of the novel parts (exits, three way decision, asymmetric cost table, segmented runtime).
- The cost table and one worked example: lambda = 10, a wrong act costs 10, a reject costs 1.
- The test suite passing on the laptop, including "exit at layer 2 never invokes later segments" (a live demo of the call counter).
- The segment file sizes at real model size (8.3 MB total int8, random weights) and the conversion working.
- The mock actuator demo (`tests/test_pipeline.py`): a spoken-intent stand in produces the exact MQTT payload, bad commands are blocked.
- Be clear that there are no accuracy or Pi latency numbers yet, and give the plan for getting them (steps above).
