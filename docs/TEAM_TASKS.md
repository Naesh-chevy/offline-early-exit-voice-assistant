# Team tasks

Project: offline early exit voice assistant (ECE 690OD). Repo: https://github.com/Naesh-chevy/offline-early-exit-voice-assistant
Trained models: https://github.com/Naesh-chevy/offline-early-exit-voice-assistant/releases/tag/v0.1-models

| Role | Who | Focus |
|---|---|---|
| A | Jaideep | Raspberry Pi, hardware, on device tests and measurements |
| B | Panashe | Models and experiments (training side) |
| C | Zoya | Wake word, own recordings, live voice testing, tests and CI |

Deadlines: measurements and ablations by Nov 27, paper draft, demo video and presentation by Dec 4,
final repo and paper Dec 9 to 14.

## How we work in git

- Never commit straight to `main`. Make a branch, commit, push the branch, open a pull request.
- Branch names: `pi-setup`, `wakeword-sweep`, `bench-exits`, and so on. One topic per branch.
- Every pull request needs one review from a teammate before it is merged. The author merges it.
- Commit messages are short and lowercase, like `add mqtt listener entry point`.
- Before every commit: `git add -n .`, read the list, then `git add`. Never commit models, datasets, `.venv`, passwords or keys.
- Set your email to your GitHub noreply address if you want to keep your real one private.
- No `git push --force`.
- Run `python -m pytest -m "not device and not gpu"` before opening a pull request. Tests that need the Pi are marked `device`.
- Every measurement script writes a CSV to `results/<stamp>_<name>/` with a header row and the git commit hash.
  Commit the CSVs. Never type a number into a result by hand. If something was not measured, write "not measured".
- Negative results are fine and valued. Write them down and report them.
- Each person adds a few plain sentences to `docs/notes.md` about the code they wrote and why.
- You must be able to explain every line you submit. If you use an AI tool for your part, disclose it in your own assignment write up.

## Role A: Jaideep, Raspberry Pi and device testing

Goal: show on a real Raspberry Pi 4B that the assistant listens on a microphone, wakes on "marvin", understands
light commands and switches an LED. Then measure latency, memory and energy and compare with the baseline.
The course says every result has to be measured on the target device, so this role is on the critical path.

### A1. Parts (check what you already have)

- Raspberry Pi 4B, 5 V 3 A USB-C power supply, microSD card (32 GB or more), heatsink or fan
- USB microphone that records at 16 kHz
- LED strip and its own power supply (do not power the strip from the Pi pins)
- Logic level N channel MOSFET (for example IRLZ44N), a 220 ohm resistor, a 10 kohm resistor
- Breadboard and jumper wires
- Optional, for energy: INA226 current sensor module, and a way to put it in series with the 5 V supply (USB-C breakout board)
- Fallback if the INA226 is too fiddly: an inline USB-C power meter. Say clearly in the report that the numbers came from it.

Order anything missing now. Delivery time is the biggest risk.

### A2. Wiring (GPIO numbers are BCM, they match `configs/device.yaml`)

- GPIO 18 (physical pin 12) to the MOSFET gate through the 220 ohm resistor
- 10 kohm between the gate and ground
- MOSFET source to ground, and connect that ground to the Pi ground (physical pin 6) and the strip supply ground
- MOSFET drain to the LED strip negative wire. The strip positive wire goes to the strip supply plus.
- GPIO 24 (physical pin 18) is the marker pin. It goes high during inference. Only needed for the energy logger.
- INA226 (optional): SDA to GPIO 2 (pin 3), SCL to GPIO 3 (pin 5), VCC to 3.3 V (pin 1), ground to pin 9.
  Put its shunt in series with the 5 V supply. The address is 0x40.

Take a clear photo and draw a simple wiring diagram. They go in the paper.

### A3. Set up the Pi

1. Flash Raspberry Pi OS 64 bit with Raspberry Pi Imager. Set hostname, user, Wi-Fi and enable SSH.
2. On the Pi:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git unzip python3-venv python3-pip libportaudio2 mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

3. Get the code and install the device requirements:

```bash
git clone https://github.com/Naesh-chevy/offline-early-exit-voice-assistant.git
cd offline-early-exit-voice-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-device.txt
```

If `ai-edge-litert` does not install on the Pi, tell Panashe right away. It is the one package we have not tried on the Pi.

4. Download the models from the release page and unzip them. `pi_models.zip` goes into `export/`:

```bash
wget https://github.com/Naesh-chevy/offline-early-exit-voice-assistant/releases/download/v0.1-models/pi_models.zip
unzip pi_models.zip -d export
```

5. The pipeline also needs the label file `fsc_labels.json` (put it in `data/prepared/`). Panashe will add it to the release.
6. The pipeline reads segments with `--seg_dir`. The device tests expect the segments directly in `export/`.
   For the tests, copy the main model there: `cp export/lam10/seg*_int8.tflite export/`
7. Find the microphone. List devices with `python -c "import sounddevice as s; print(s.query_devices())"` and set
   `mic.device_index` in `configs/device.yaml` if the default is not the USB mic.
8. Before measuring, set the CPU governor and check the temperature:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
vcgencmd measure_temp
vcgencmd get_throttled
```

`get_throttled` should be `0x0`. If not, the Pi is under voltage or too hot, so fix that before measuring.

### A4. Run the device tests

```bash
python -m pytest -m device -v
```

There are four: the mic records one second, the light toggles, the naive baseline runs on the device, and exit at
layer 2 is faster than running all layers. Report what fails and why. Do not change a test just to make it pass.

### A5. Code you write (small pull requests, one each)

1. **Listener entry point.** `src/actuators/gpio.py` has `serve_mqtt(...)` but nothing starts it. Add an
   `if __name__ == "__main__":` block that loads `configs/device.yaml` and calls
   `serve_mqtt(cfg["actuators"]["gpio"], cfg["actuators"]["mqtt"])`. Test it with
   `mosquitto_pub -t home/kitchen_lights/set -m '{"action":"on","value":null}'` and the same with `"off"`.
2. **systemd services** for the listener and the pipeline, so both start on boot. Put the files in `docs/systemd/`.
3. **`src/measure/bench_exits.py`.** The existing `bench.py` times one model with random input. We also need the
   real early exit measurement. For N real clips (Panashe will put a 200 clip FSC test subset in the release),
   run the segmented encoder with the policy and record the exit, the encoder time per clip and whether the
   command was right. Do it for: early exit with reject (the thresholds in `configs/train_encoder.yaml`),
   full encoder only, and the naive fp32 model. Use `SegmentedEncoder`, `run_policy` and `measure_latency` as they are.
4. **Wake word on the Pi.** Record 1 to 2 hours of normal room noise with the Pi mic to a wav file for Zoya.
   Then check how often the wake word fires in a day of normal use, and note it.

### A6. Measurements to produce (CSV for each, 3 repeated sessions)

- `python -m src.measure.bench --model export/naive_fp32.tflite --run_name naive_fp32`
- The same for `export/lam10/seg1_int8.tflite`, `seg2_int8.tflite`, `seg3_int8.tflite`, for lambda 5, and for `export/wakeword_int8.tflite`
- The QUADS graph `quads_8bit_fp32.tflite` once Panashe has trained it
- `bench_exits.py` results for lambda 10 and lambda 5
- Peak memory (RSS) is included in the bench output. Energy only if the meter is wired, otherwise it says "not measured".
- Note the temperature and governor for each run. Wait for the Pi to cool between runs.

### A7. Live demo

Run `python -m src.runtime.pipeline --seg_dir export/lam10`, say "marvin, turn on the kitchen lights", and watch the
LED. Record a demo video that shows the live mic, the LED, and the printed exit layer.

### A8. What Role A hands in for the paper

Device results table (latency p50 and p90, peak RSS, energy if measured) for the naive model, the early exit model
and the full encoder, the wiring diagram and photo, the demo video, and a short list of problems you hit.

## Role C: Zoya, wake word, own recordings, live testing, tests and CI

Goal: make the wake word and data side solid, test the whole system with our own voices, and keep the code healthy.

### C1. Laptop setup

```bash
git clone https://github.com/Naesh-chevy/offline-early-exit-voice-assistant.git
cd offline-early-exit-voice-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -m "not device and not gpu"
```

(Use `source .venv/bin/activate` on Linux or Mac. Python 3.11 is the target.)
Models are in the release page. For wake word work you also need Speech Commands v2:
`python data/download.py sc` then `python data/prepare.py sc`. FSC needs the five tar files from
https://zenodo.org/records/14722453 in `data/raw/fsc_zenodo/`, then `python data/prepare.py fsc`.

### C2. Wake word

Read `src/wakeword/train.py` and `configs/train_wakeword.yaml`. The model reaches 99.35 percent accuracy and 7.2
percent false rejects, with about 90 false accepts per hour. That last number comes from only about 80 seconds
of noise, so it is a rough upper bound.

1. **Threshold sweep.** Write `src/wakeword/sweep.py`. Load the int8 model with `Segment`, compute the wake
   probabilities and sweep the threshold from 0.1 to 0.95. For each value, report the false reject rate on the test
   wake clips and the false accepts per hour on held out noise (reuse `false_accepts_per_hour` and `metrics`).
   Write a CSV and a plot of false rejects against false accepts per hour. Pick an operating point and explain why.
2. **Real noise.** Jaideep records 1 to 2 hours of room noise on the Pi mic. Run it through the sweep. That is the
   honest false accept number.
3. **Own voices.** Each of us records about 30 "marvin" clips at different distances and with some background noise.
   Save them in `data/own/wake/` (this folder is not committed if it is large). Measure the false reject rate on them.
4. **If false accepts stay high:** try more noise only negatives, hard negative words that sound like marvin,
   or more epochs. Retraining runs on Colab (`gpu_package/colab_training.ipynb`, cell 7). The free Colab GPU is enough.
   Never overwrite the released model. Name new checkpoints clearly.

### C3. Own recordings for the assistant commands

1. Write `data/record_own.py`. It prompts a phrase, records a 4 second clip at 16 kHz mono with `sounddevice`,
   saves `data/own/<speaker>/<number>.wav`, and appends a row to `data/prepared/fsc_own.csv`.
   Use the same columns as `data/prepared/fsc_test.csv`.
2. Phrases to record (FSC style, 10 clips each per person):
   - "turn on the lights", "turn off the lights" (no room means the living room)
   - "turn on the kitchen lights", "turn off the kitchen lights"
   - "turn on the bedroom lights", "turn off the bedroom lights"
   - Variants: "switch on the lights", "lights on", "turn the lights off"
   - Things that must be rejected or ignored: "turn on the washroom lights" (no device), "turn up the heat", "play music"
   - "turn on the living room lights", which FSC never contains. Expect it to be weaker, and report it as it is.
3. Add a `--splits` option to `src/encoder/eval_exits.py` so it can also score the `own` split. Report accuracy per
   speaker and per phrase and compare with the FSC test numbers. A gap between our voices and FSC is a valid result.
4. Never edit labels to make numbers look better.

### C4. Tests and CI

1. Add `.github/workflows/tests.yml`: run on pull requests and pushes to `main`, Python 3.11, install CPU torch from
   `https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`, then
   `python -m pytest -m "not device and not gpu" -q`. Add `libportaudio2` to the runner. Cache pip so it runs faster.
   Add a status badge to the README.
2. Add pipeline tests with the mock actuator for edge cases: silence after the wake word, empty audio, a
   rejected command, and a command for a device that does not exist. See `tests/test_pipeline.py` for the style.
3. Check the README setup steps on a fresh clone and fix anything that does not work.

### C5. What Role C hands in for the paper

Wake word results and the sweep plot, the live voice evaluation table, a short description of the own dataset
(speakers, phrases, counts), and the setup and CI notes.

## Role B: Panashe, models and experiments

- Train the QUADS style prior method on Colab at 8 bit and 4 bit, and report the results.
- Run 3 seeds of the exit heads. Add per clip predictions and a paired test (bootstrap or McNemar) so the comparison
  between early exit and the naive baseline has confidence intervals.
- No distillation ablation, and optionally longer exit head training and SLURP.
- Plots: wrong actions against lambda, exit counts, accuracy against layers saved.
- Add to the release: `fsc_labels.json`, a 200 clip FSC test subset with its manifest, and `quads_8bit_fp32.tflite` when it exists.
- Review pull requests from both teammates.

## Handoffs

| From | To | What | When |
|---|---|---|---|
| Panashe | Jaideep | `fsc_labels.json` and the 200 clip test subset in the release | Nov 20 |
| Jaideep | Zoya | 1 to 2 hours of room noise from the Pi mic | Nov 25 |
| Everyone | Zoya | About 30 "marvin" clips and the command phrases, per person | Nov 24 |
| Panashe | Jaideep | `quads_8bit_fp32.tflite` and its results | after the QUADS run |

## Timeline

- By Nov 22: Jaideep orders parts, flashes the Pi and runs the setup. Zoya has the laptop setup done and a CI pull request open.
  Panashe starts the QUADS run and uploads the extra release files.
- By Nov 27: Wiring works, the listener starts, `pytest -m device` has been run, naive and int8 bench CSVs exist.
  The wake word sweep and the first own recordings are done. Seeds and confidence intervals are finished.
- By Dec 4: All measurements are done, plots are made, paper sections are drafted, demo video is recorded.
- Dec 5 to 8: Polish, fix gaps, final repo check.

Paper sections: A writes the device and measurement parts, B the model, results and ablations, C the dataset and wake word parts.
