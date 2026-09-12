"""
Step 3: Generate New Music + Save as MIDI
--------------------------------------------
Loads the trained model + mappings, seeds it with a random sequence from
the training data, then repeatedly predicts the "next" note/chord and
feeds it back in (autoregressive generation). Converts the resulting
sequence back into a music21 stream and writes it out as a .mid file.

Output: output/generated_music.mid
"""

import pickle
import numpy as np
from tensorflow import keras
from music21 import stream, note, chord, duration

# ---------------- CONFIG ----------------
MODEL_PATH = "model/music_lstm.keras"
MAPPINGS_PATH = "model/mappings.pkl"
NOTES_PATH = "data/notes.pkl"
OUTPUT_MIDI = "output/generated_music.mid"
GENERATE_LENGTH = 200     # number of notes/chords to generate
TEMPERATURE = 1.0          # >1 = more random/creative, <1 = more conservative/repetitive
# -----------------------------------------


def sample_with_temperature(probabilities, temperature):
    """Sample an index from a probability distribution, adjusted by temperature."""
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-9) / temperature
    exp_preds = np.exp(probabilities)
    probabilities = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probabilities), p=probabilities)


def generate_notes(model, seed_sequence, note_to_int, int_to_note, vocab_size, length, temperature):
    pattern = list(seed_sequence)
    generated = []

    for _ in range(length):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
        prediction = model.predict(input_seq, verbose=0)[0]
        idx = sample_with_temperature(prediction, temperature)
        result = int_to_note[idx]
        generated.append(result)

        pattern.append(idx)
        pattern = pattern[1:]  # slide the window forward

    return generated


def events_to_midi(events, output_path):
    """Turn a list of event strings back into a music21 stream and save as MIDI."""
    output_stream = stream.Stream()
    for event in events:
        if ('.' in event) or event.isdigit():
            # It's a chord, stored as pitch-class integers joined by dots
            notes_in_chord = event.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.duration = duration.Duration(0.5)
            output_stream.append(new_chord)
        else:
            # It's a single note, e.g. 'C4'
            new_note = note.Note(event)
            new_note.duration = duration.Duration(0.5)
            output_stream.append(new_note)

    output_stream.write('midi', fp=output_path)


def main():
    import os
    os.makedirs("output", exist_ok=True)

    print("Loading model and mappings...")
    model = keras.models.load_model(MODEL_PATH)
    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)
    note_to_int = mappings["note_to_int"]
    int_to_note = mappings["int_to_note"]
    vocab_size = mappings["vocab_size"]
    seq_length = mappings["seq_length"]

    with open(NOTES_PATH, "rb") as f:
        events = pickle.load(f)

    # Pick a random starting seed from the training data
    start_idx = np.random.randint(0, len(events) - seq_length - 1)
    seed_events = events[start_idx:start_idx + seq_length]
    seed_sequence = [note_to_int[n] for n in seed_events]

    print(f"Generating {GENERATE_LENGTH} notes/chords (temperature={TEMPERATURE})...")
    generated_events = generate_notes(
        model, seed_sequence, note_to_int, int_to_note,
        vocab_size, GENERATE_LENGTH, TEMPERATURE
    )

    print("Converting to MIDI...")
    events_to_midi(generated_events, OUTPUT_MIDI)
    print(f"Saved generated music to {OUTPUT_MIDI}")


if __name__ == "__main__":
    main()