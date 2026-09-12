# 🎵 AI Music Generation using LSTM

A deep learning project that generates original music by training a stacked LSTM (Long Short-Term Memory) neural network on classical music (Bach chorales) and produces new note/chord sequences saved as playable MIDI files.

> Built as part of the **CodeAlpha Internship — Music Generation with AI** task.

---

## 📌 Overview

This project demonstrates end-to-end music generation using deep learning:

1. **Data Collection** — Classical MIDI-equivalent scores (Bach chorales) sourced from `music21`'s built-in public-domain corpus.
2. **Preprocessing** — Converts musical notes and chords into a numerical sequence format suitable for training.
3. **Model** — A 2-layer stacked LSTM network learns patterns in the note sequences.
4. **Training** — The model learns to predict the next note/chord given the previous 40.
5. **Generation** — The trained model creates new, original note sequences.
6. **Output** — Sequences are converted back into a standard `.mid` file that can be played in any MIDI player, DAW, or converted to audio.

---

## 🏗️ Architecture

```
Raw Music (Bach Chorales)
        │
        ▼
┌───────────────────┐
│  music21 parsing   │  → extracts notes & chords
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Sequence Encoding  │  → notes/chords → integers
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   LSTM(256) x2     │
│   + Dropout        │
│   + Dense layers   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Autoregressive     │  → predict next note, repeat
│ Generation         │
└───────────────────┘
        │
        ▼
   generated_music.mid
```

**Model summary:**

| Layer | Output Shape | Params |
|---|---|---|
| LSTM (256 units, return_sequences) | (40, 256) | 264,192 |
| Dropout (0.3) | (40, 256) | 0 |
| LSTM (256 units) | (256,) | 525,312 |
| Dense (256, ReLU) | (256,) | 65,792 |
| Dropout (0.3) | (256,) | 0 |
| Dense (vocab_size, Softmax) | (vocab_size,) | 17,219 |

**Total trainable params:** ~872K

---

## 📂 Project Structure

```
Task_music_generation/
├── prepare_data.py         # Step 1: Data collection + preprocessing
├── train_model.py          # Step 2: Build & train the LSTM
├── generate_music.py       # Step 3: Generate music + export MIDI
├── requirements.txt        # Python dependencies
└── README.md
```

Running the scripts in order will automatically create these folders alongside them:

```
data/
└── notes.pkl               # Preprocessed note/chord sequences (created by prepare_data.py)

model/
├── music_lstm.keras        # Trained model weights (created by train_model.py)
└── mappings.pkl            # note <-> integer vocabulary lookup

output/
└── generated_music.mid     # Final generated music (created by generate_music.py)
```

---

## ⚙️ Tech Stack

- **Python 3.12**
- **[music21](https://web.mit.edu/music21/)** — music theory & MIDI parsing toolkit
- **TensorFlow / Keras** — deep learning framework
- **NumPy** — numerical processing

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
```bash
python3 prepare_data.py
```
This extracts note/chord sequences from 60 Bach chorales (built into `music21`, no download required) and saves them to `data/notes.pkl`.

### 3. Train the model
```bash
python3 train_model.py
```
Trains a stacked LSTM for 60 epochs (configurable). Saves the trained model to `model/music_lstm.keras`.

> 💡 Training is CPU-friendly but faster on GPU (e.g. Google Colab). ~30-40 min on CPU for 60 epochs with this dataset size.

### 4. Generate new music
```bash
python3 generate_music.py
```
Loads the trained model, generates 200 new notes/chords, and saves them as `output/generated_music.mid`.

### 5. Play the result
Open `generated_music.mid` in any MIDI player, DAW (GarageBand, FL Studio, Ableton), or convert it to audio using a free SoundFont + tool like `fluidsynth` / `timidity`.

---

## 🎛️ Configuration Options

| File | Parameter | What it does |
|---|---|---|
| `prepare_data.py` | `NUM_PIECES` | How many pieces to train on (more = richer data) |
| `prepare_data.py` | `USE_LOCAL_MIDI` | Set `True` to use your own `.mid` files instead |
| `train_model.py` | `SEQ_LENGTH` | Context window size the model looks at |
| `train_model.py` | `EPOCHS` | Training iterations |
| `train_model.py` | `LSTM_UNITS` | Model capacity |
| `generate_music.py` | `GENERATE_LENGTH` | Number of notes to generate |
| `generate_music.py` | `TEMPERATURE` | Randomness/creativity of generation (>1 = more random, <1 = more predictable) |

---

## 🧠 How It Works (Explanation)

**1. Preprocessing:**
Each single note is represented by its pitch name (e.g. `C4`), and each chord is represented by its normal-order pitch classes joined with dots (e.g. `4.7.11`). This turns music into a categorical sequence — conceptually the same problem as text/character generation.

**2. Sequence framing:**
The model is trained the same way a text-generation model is: given the previous 40 notes/chords, predict the next one. This creates a supervised learning problem from unlabeled music.

**3. Model:**
A stacked LSTM is used because it can capture both short-term note-to-note transitions and longer melodic/harmonic patterns across the 40-step window. Dropout layers reduce overfitting.

**4. Generation:**
Generation is autoregressive: the model predicts one note, that note is appended to the input, the oldest note is dropped, and the process repeats. A temperature parameter controls how "safe" vs. "creative" the sampling from the output probability distribution is.

**5. Output:**
The generated integer sequence is mapped back to note/chord names and reconstructed into a `music21` stream, which is then exported as a standard MIDI file.

---

## 📊 Results

- Dataset: 60 Bach chorales → 9,322 note/chord events, vocabulary of 67 unique tokens
- Training loss: dropped from ~3.64 to ~2.39 over 30 epochs (further improves with more epochs)
- Output: a valid, playable 200-event MIDI file

---

## 🔮 Future Improvements

- Train on a larger/more diverse dataset (jazz, pop, multiple composers)
- Add note duration and velocity as additional features (currently fixed duration)
- Try a GAN-based approach (e.g. MuseGAN-style) for comparison
- Add attention layers or a Transformer-based architecture
- Convert output to audio (WAV/MP3) automatically using a SoundFont synthesizer

---

## 📄 License

This project uses the `music21` corpus (public domain Bach chorales) for training data. Free to use and modify for educational purposes.