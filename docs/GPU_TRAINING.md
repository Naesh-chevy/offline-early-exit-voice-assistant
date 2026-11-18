# GPU training instructions

Hi! Thanks for helping. This trains the Whisper encoder models for an offline voice assistant project (ECE 690OD).
You only need an NVIDIA GPU, Python 3.10 or newer, about 15 GB of free disk, and internet. Nothing here touches your files outside this folder.

Rough time: unknown until you test. Do the 1 epoch test first (step 4) and tell us how long it took.

Tip: if the run might get interrupted, add `--resume` to the `distill` and `train_exits` commands. It continues from the last saved epoch. (There is also `colab_training.ipynb` for Google Colab.)

## 1. Unpack and set up

Unzip `gpu_training_package.zip` into a new folder, put the `data_files/` folder you were sent inside it, and open a terminal there.

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on Linux/Mac, `.venv\Scripts\activate` on Windows). Then install PyTorch with CUDA.
Get the exact command for your CUDA version from https://pytorch.org/get-started/locally/ and run it, then:

```bash
pip install numpy pyyaml transformers soundfile tqdm psutil tensorflow ai-edge-litert
```

Check the GPU is seen:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True`. If it prints `False`, stop and tell us.

## 2. Put the dataset files in place

You were sent a folder with the code zip and a `data_files/` folder. The data files are the two datasets we need, already downloaded:
`speech_commands_v0.02.tar.gz` (2.3 GB) and five `fluentspeechcommands_*.tar` files (2.3 GB).

Check nothing was damaged in transfer (compare with `checksums.md5` in the same folder):

```bash
cd data_files
md5sum -c checksums.md5
cd ..
```

On Windows without `md5sum`, use `certutil -hashfile <file> MD5` and compare by eye.

Copy the files into the project folder you unzipped:

```bash
mkdir -p data/raw/fsc_zenodo data/raw/speech_commands_v2
cp data_files/fluentspeechcommands_*.tar data/raw/fsc_zenodo/
tar -xzf data_files/speech_commands_v0.02.tar.gz -C data/raw/speech_commands_v2
```

(`tar` works on Linux, Mac and Windows 10 or newer. On Windows PowerShell, create the folders by hand if `mkdir -p` does not work.)

After extracting, `data/raw/speech_commands_v2/` should contain folders like `marvin`, `yes`, `_background_noise_` and the files `validation_list.txt` and `testing_list.txt`.

If you do not have the `data_files/` folder: the Speech Commands v2 download is `python data/download.py sc`, and the FSC files are the five tars from https://zenodo.org/records/14722453 (put them in `data/raw/fsc_zenodo/`).

## 3. Prepare the data

```bash
python data/prepare.py sc
```

It should print `sc train: 84843 clips, 1710 wake`, `sc valid: 9981 clips, 195 wake`, `sc test: 11005 clips, 195 wake`.

```bash
python data/prepare.py fsc
```

It should print `fsc train: 23132 clips`, `fsc valid: 3118 clips`, `fsc test: 3793 clips`. If the numbers differ, stop and tell us.

## 4. Test with 1 epoch first (important)

A few training steps were tested on CPU with the real data, but no full epoch has been run. A 1 epoch test finds problems cheaply on a real GPU. Time each step and write down the times.

```bash
python -m src.encoder.distill --stage teacher --epochs 1 2>&1 | tee teacher_test.log
```

If this crashes, send us the full error text. The first run downloads Whisper weights from Hugging Face (about 300 MB).
If you get an out of memory error, lower `batch_size` under `train:` in `configs/train_encoder.yaml` (for example 16) and retry.

## 5. Real training

### 5a. Wake word model (independent of the others)

```bash
python -m src.wakeword.train --run_name wakeword 2>&1 | tee wakeword.log
```

It trains the small wake word model for the word "marvin", does int8 quantization aware training, and writes `export/wakeword_int8.tflite` and a results CSV. Check the end of the log: it prints accuracy, false reject rate and false accepts per hour for fp32 and int8. If it crashes, send us the error.

### 5b. Whisper encoder models, in this order

Each step needs the one before it. Keep the logs with `tee`. You can lower the number of epochs with `--epochs 10` if time is short; tell us what you used.

```bash
python -m src.encoder.distill --stage teacher 2>&1 | tee teacher.log
```

```bash
python -m src.encoder.distill --stage student 2>&1 | tee student.log
```

```bash
python -m src.encoder.distill --stage student --no_distill 2>&1 | tee student_nodistill.log
```

Now the exit heads. Each command trains four models (lambda 1, 2, 5, 10):

```bash
python -m src.encoder.train_exits --encoder checkpoints/student/encoder.pt --tag distill 2>&1 | tee exits_distill.log
```

```bash
python -m src.encoder.train_exits --encoder checkpoints/student_nodistill/encoder.pt --tag nodistill 2>&1 | tee exits_nodistill.log
```

Finally the naive FP32 baseline (the comparison model). It also writes `export/naive_fp32.tflite`:

```bash
python -m src.baselines.naive 2>&1 | tee naive.log
```

Optional last check (should mostly pass, tell us the output either way):

```bash
python -m pytest -m gpu -q
```

## 6. Send back

Please zip and send us these:
- the whole `checkpoints/` folder
- the whole `export/` folder (it has the wake word model)
- the whole `results/` folder
- all the `.log` files
- the epoch counts you used and roughly how long each step took

Thank you!
