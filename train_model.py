"""
Step 2: Build + Train the LSTM Model
--------------------------------------
Loads data/notes.pkl, turns it into fixed-length input sequences,
builds a stacked LSTM, and trains it to predict the next note/chord
given the previous SEQ_LENGTH events.

Output:
  model/music_lstm.keras   (trained model)
  model/mappings.pkl       (note<->int lookup tables, needed for generation)
"""

import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
SEQ_LENGTH = 40          # how many previous events the model looks at
EPOCHS = 60  # increase for a more sophisticated model if you have time/GPU
BATCH_SIZE = 64
LSTM_UNITS = 256
NOTES_PATH = "data/notes.pkl"
MODEL_OUT = "model/music_lstm.keras"
MAPPINGS_OUT = "model/mappings.pkl"
# -----------------------------------------


def load_events():
    with open(NOTES_PATH, "rb") as f:
        return pickle.load(f)


def build_sequences(events, seq_length):
    pitch_names = sorted(set(events))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}
    int_to_note = {i: n for n, i in note_to_int.items()}
    vocab_size = len(pitch_names)

    network_input = []
    network_output = []
    for i in range(len(events) - seq_length):
        seq_in = events[i:i + seq_length]
        seq_out = events[i + seq_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    X = np.reshape(network_input, (n_patterns, seq_length, 1))
    X = X / float(vocab_size)  # normalize
    y = keras.utils.to_categorical(network_output, num_classes=vocab_size)

    return X, y, note_to_int, int_to_note, vocab_size


def build_model(seq_length, vocab_size):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(LSTM_UNITS, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main():
    import os
    os.makedirs("model", exist_ok=True)

    events = load_events()
    print(f"Loaded {len(events)} events.")

    X, y, note_to_int, int_to_note, vocab_size = build_sequences(events, SEQ_LENGTH)
    print(f"Built {X.shape[0]} training sequences. Vocab size: {vocab_size}")

    with open(MAPPINGS_OUT, "wb") as f:
        pickle.dump({
            "note_to_int": note_to_int,
            "int_to_note": int_to_note,
            "vocab_size": vocab_size,
            "seq_length": SEQ_LENGTH,
        }, f)

    model = build_model(SEQ_LENGTH, vocab_size)
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="loss", save_best_only=True, verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)

    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
    )

    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()