# Notes (study these before the presentation)

## Features (`src/audio/features.py`)
Audio is 16 kHz mono. We make an 80 bin log mel spectrogram the same way Whisper does (hann window, 400 point fft,
hop 160, log10, clip to max minus 8, then scale). The clip is padded or cropped first, so 4 s gives 400 frames and
1 s gives 100 frames. It is plain numpy so the same code runs on the laptop and the Pi.

## Mic (`src/audio/mic.py`)
A ring buffer keeps the newest 2 s of audio and overwrites the old audio. The VAD says "speech" when the frame
rms is above a threshold, and keeps saying it for a few frames after (hangover) so we do not cut words off.

## Encoder (`src/encoder/student.py`)
Our own Whisper style encoder: two convs, fixed sinusoidal position embeddings, then pre layer norm transformer
blocks. Names match Hugging Face so pretrained tiny and base weights load. Whisper wants 30 s (1500 positions) but
we only feed 4 s, so we only keep the first 200 positions. Layers are numbered from 1 to match the exit numbers.

## Naive baseline (`src/baselines/naive.py`)
Full 4 layer FP32 tiny encoder, mean pool over time, three linear classifiers (action, object, location). It uses
the same 4 s window as the real system, so the comparison is about exits and int8, not input length.

## TFLite export (`src/export/tf_encoder.py`)
PyTorch to TFLite tools like ai-edge-torch do not run on Windows, so we rebuilt the encoder pieces by hand in plain
TF ops, with the weights copied from the PyTorch state dict. Tests check that PyTorch and TFLite agree. Conv uses
explicit padding because TF "same" padding with stride 2 pads unevenly and would not match PyTorch.

## Measurement (`src/measure/`)
Latency: warmup runs, then timed runs, we report mean, p50, p90, p99. Memory: a thread polls RSS while the model
runs and we keep the peak. Energy: if there is no INA226 we write "not measured", never a guess. Every CSV gets the
git hash and device info (governor, temperature on the Pi).

## Actuators (`src/actuators/`)
The registry lists devices, allowed actions and value ranges (thermostat 16 to 30). The client checks every command
against it before publishing to MQTT, so a bad command never leaves the program. The mock broker has the same
`publish` call so tests need no network. BLE is not written yet.

## Wake word (`src/wakeword/`)
Small DS-CNN (depthwise separable convs). Training order: fp32 with batchnorm, fold batchnorm into the convs, then
QAT fine tuning. QAT here is our own: weights and activations are rounded to the int8 grid in the forward pass, and
the backward pass pretends the rounding did not happen (straight through estimator). Then TFLite converts to a full
int8 file using calibration data. Caveat: TFLite picks its own activation scales during conversion, so the QAT
scales are not copied over exactly. QAT just makes the weights tolerant to rounding.

## Distillation (`src/encoder/distill.py`)
Teacher is Whisper base fine tuned on FSC. Student loss = task cross entropy + KL between softened logits (temperature
T, multiplied by T squared so the gradient size does not depend on T) + MSE between projected student hidden states
and teacher hidden states at matched layers (student 2, 3, 4 with teacher 2, 4, 6). The no distillation run uses the
same seed and budget but only the task loss.

## Exits, loss and policy (the novel part)
- `exits.py`: after layer 2, 3 and 4, mean pool the hidden state and run small heads: action, object, location,
  slot (thermostat setpoint class) and a decision head with three outputs ACT, CONTINUE, REJECT. The last exit gets
  its CONTINUE logit set to a huge negative number because there is nothing left to run.
- `loss.py`: at each exit, cost of ACT is 0 if the intent is right and lambda if wrong, REJECT costs 1, CONTINUE costs
  a small fixed price (0.2). "Right" is the product of the probabilities the intent heads give to the true labels,
  so it is soft and gradients reach the intent heads. The decision loss is the expected cost: sum of decision
  probability times cost. With lambda = 1 a wrong act and a reject cost the same. With lambda = 10 a wrong act is
  ten times worse than asking the user to repeat. This is also added to normal cross entropy so the intent heads
  learn early. Exit losses are summed with weights, and the last exit teaches the earlier ones with a KL term.
- `policy.py`: go through exits in order. ACT if p(ACT) is over the exit threshold. Otherwise, at the last exit,
  REJECT. At earlier exits, REJECT if p(REJECT) is over its threshold, else CONTINUE. The function that runs an
  exit is only called when we get there, which is what makes the later segments never run.

## Segmented runtime (`src/export/segments.py`, `src/runtime/segmented.py`)
The encoder is exported as three int8 TFLite files: frontend plus layers 1 and 2 plus exit 1 heads, layer 3 plus exit
2 heads, layer 4 plus exit 3 heads. Python hands the hidden state from one file to the next (dequantize then
requantize, because each file has its own scale). `SegmentedEncoder.exit_fn(k)` runs segment k only when the policy
asks, and each segment counts its calls so a test can prove the later ones never ran. The CONTINUE mask for the
last exit is applied in Python after the model, not inside it, because a -10000 logit would ruin the int8 range.

## Pipeline (`src/runtime/pipeline.py`)
Wake word on the last 1 s of the ring buffer every 0.25 s. When it fires, record up to 4 s until the VAD says speech
stopped, make features, run the policy, map labels to a command, check the registry, send over MQTT.

## Resume saving (`src/encoder/ckpt.py`)
Long runs on Colab can get cut off. After every epoch we save the model weights, the optimizer state and the number of finished epochs. With `--resume` the script loads that file and carries on from the next epoch, so a disconnect costs at most one epoch. The file is written to a temp name and then renamed, so a crash while saving cannot break the last good copy. A finished lambda in `train_exits` just loads its saved state and has nothing left to train.

## Scope: lights only (changed 2026-11-05)
The registry now has three devices: kitchen, bedroom and living room lights, each with on and off. Everything else the model can hear (music, heat, shoes, curtains) maps to no device, so the registry rejects it and nothing is sent. FSC has no "living room", and its "turn on the lights" has location "none". So "no room said" maps to the living room (the default room). Saying "living room" out loud is not in FSC, so the model has not heard it. The washroom has no light in the registry, so it is rejected.


## Wake word first result and fix (2026-11-06)
First GPU run: test accuracy 98.76% fp32 and 98.68% int8, model 70.5 KB fp32 and 33.2 KB int8, false reject 6.7%. But false accepts on held out background noise were about 1577 per hour (fp32) and 2118 per hour (int8), roughly 11% of noise windows. Cause: the "other" class was only spoken words, so the model never saw noise only or silent windows as negatives, and the live pipeline checks exactly those windows every 0.25 s. Fix: `noise_neg_frac` adds noise only and near silence windows as label 0. The first run's CSV is kept as a negative result. Caveat on the metric: the held out noise is only the last 20% of six short noise files (a bit over a minute), windows overlap, so the per hour number is a rough upper bound.

Second run, with noise only negatives (`noise_neg_frac` 0.25): test accuracy 99.34% fp32 and 99.35% int8, false reject 7.2% for both, int8 still 33.2 KB. False accepts per hour on held out noise dropped from 1577 to 135 (fp32) and from 2118 to 90 (int8). The held out noise is only about 80 s, so these are 3 and 2 windows. The fp32 vs int8 difference is not meaningful, and 90 to 135 per hour is still a lot. Next: sweep the threshold, and measure false accepts on a long recording of the real room with the Pi microphone. Results folders: `20261106_073323_wakeword` (first run) and `20261106_080645_wakeword` (second run).

## Exit evaluation on the real model (2026-11-06, `src/encoder/eval_exits.py`)
Thresholds are picked on valid (grid of act thresholds, cost = lambda * false action rate + reject rate) and then reported on test. Valid is much harder than test (valid has only 10 speakers): the naive baseline gets 94.0% intent on valid but 98.8% on test, so do not compare valid and test numbers with each other. The placeholder thresholds in the config were far too loose, the tuned ones are about 0.98 to 0.99.

Test set (3793 clips), false action rate / reject rate / mean exit (1 = after layer 2, 3 = after layer 4):
- naive FP32 baseline, no exits, no reject: 1.24% wrong, mean exit 3 (all layers)
- lambda 1, early exit: 2.53% / 0.03% / 1.00
- lambda 2, early exit: 2.14% / 0.08% / 1.07
- lambda 5, early exit: 1.27% / 1.13% / 2.58 (no reject: 1.48% / 0 / 2.58)
- lambda 10, early exit: 0.92% / 0.66% / 2.74 (no reject: 1.16% / 0 / 2.56)
- lambda 10, full encoder only (exit 3, never stop early): 0.90% / 0.74% / 3.0

So small lambda gives a big compute saving (stops at layer 2) but about 2x the wrong actions. Large lambda gives roughly the same wrong action rate as the full encoder while running about 0.3 to 0.4 fewer exits on average. Reject removes about 0.2 points of wrong actions at lambda 5 and 10, that is about 9 clips out of 3793, single seed, so it is a small effect and I would not oversell it.

int8 check on test (3793 clips): fp32 TFLite matches PyTorch to about 1e-6 in the logits. int8 segments agree with PyTorch on the full intent for 99.2 to 99.8% of clips per exit, decision head argmax agrees on 99.4 to 100%. With the same thresholds the int8 policy gives wrong action 1.29% (lambda 5) and 1.03% (lambda 10) vs 1.27% and 0.92% in fp32, a loss of 0 to 0.1 points, within noise for this test size. Size: 8.3 MB int8 vs 31.4 MB fp32 for the three segments. Latency on the Pi 4 is not measured yet.

## Prior method: QUADS style (`src/baselines/quads.py`)
QUADS (Interspeech 2025) shrinks a Whisper encoder SLU model by alternating two kinds of epochs. A distill epoch trains the student with cross entropy plus an L1 loss between pooled teacher and student encoder outputs. A quantize epoch gives every weight matrix a k-means codebook (2^bits centroids), and after every optimizer step each centroid is moved to the mean of the weights in its cluster and the weights are snapped to the centroids. The run ends with quantize epochs only. Our version uses the same idea with our whisper tiny student, our FSC finetuned whisper base teacher, three heads and 4 s input, at 8 and 4 bits. Weights stay fp32 in memory, so it saves storage but not compute: on the Pi it runs like the naive FP32 baseline, which is why we also export `quads_*bit_fp32.tflite` to time it. Three honest notes: the paper does not give the L1 weight (their repo uses 1e-6, which makes the teacher almost irrelevant, we use 0.1, a sweep is possible); in the repo's epoch lists, epochs 16 to 20 do nothing; accuracy dips after each distill epoch because the weights get re-snapped, and the next quantize epoch recovers it, so the schedule must end on quantize epochs.
