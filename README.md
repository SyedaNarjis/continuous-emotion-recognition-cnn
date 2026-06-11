# Unimodal CNN — CreativeIT IS17 Feature Set

A convolutional neural network for continuous **emotion recognition** from speech and/or motion features using the [CreativeIT dataset](https://sail.usc.edu/CreativeIT/ImprovDatabase.htm) with the [IS17 feature set](http://www.compare.openaudio.eu/).

Predicts **Activation (A)**, **Valence (V)**, or **Dominance (D)** using leave-one-session-out cross-validation. Performance is reported as **CCC**, **PCC**, and **RMSE**.

---

## Model Architecture

```
Input (window_size × n_features × 1)
  └── Conv2D(16, 1×13, ReLU) → MaxPool → BatchNorm
  └── Flatten
  └── Dense(512, ReLU) → Dropout(0.25) → BatchNorm
  └── Dense(256, ReLU) → Dropout(0.50) → BatchNorm
  └── Dense(64,  ReLU) → Dropout(0.50) → BatchNorm
  └── Dense(1)          ← regression output
Loss: MSE | Optimizer: Adam
```

---

## Requirements

| Package    | Tested version |
|------------|---------------|
| Python     | ≥ 3.9         |
| TensorFlow | ≥ 2.10        |
| Keras      | ≥ 2.10        |
| NumPy      | ≥ 1.24        |

```bash
pip install tensorflow numpy
```

---

## Project Structure

```
unimodal_cnn_cit_is17set/
├── main_unicnn.py          # Entry point — training & evaluation
├── modelcnn_tester.py      # CNN architecture
├── load_features.py        # Feature loading and windowing utilities
├── calc_scores.py          # CCC / PCC / RMSE metrics
├── write_predictions.py    # Save test predictions to CSV
├── requirements.txt
└── README.md
```

---

## Usage

```bash
# Audio + motion (default)
python main_unicnn.py \
    --audio-path  /path/to/audio_features/ \
    --motion-path /path/to/motion_features/ \
    --label-path  /path/to/labels/

# Audio only
python main_unicnn.py --no-motion --audio-path /path/to/audio_features/ ...

# Motion only
python main_unicnn.py --no-audio --motion-path /path/to/motion_features/ ...
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio-path` | `features/audio/` | Audio feature CSV directory |
| `--motion-path` | `features/motion/` | Motion feature CSV directory |
| `--label-path` | `features/labels/` | Label CSV directory |
| `--results` | `results.txt` | Output file for scores |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `200` | Mini-batch size |
| `--window-size` | `120` | Temporal window length (frames) |
| `--hop` | `20` | Stride between windows (frames) |
| `--audio / --no-audio` | on | Use audio modality |
| `--motion / --no-motion` | on | Use motion modality |

---

## Data

The features and labels follow the semicolon-delimited CSV format used by the [openSMILE](https://www.audeering.com/research/opensmile/) IS17 feature extractor:

```
instance_name ; timestamp ; feat_1 ; feat_2 ; ...
```

Data is **not included** in this repository. Please refer to the [CreativeIT dataset page](https://sail.usc.edu/CreativeIT/ImprovDatabase.htm) for access.

---

## License

MIT — see [LICENSE](LICENSE) for details.
