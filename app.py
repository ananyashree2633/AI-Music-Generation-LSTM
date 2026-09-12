"""
Streamlit Web App: AI Music Generator
----------------------------------------
A simple browser-based demo for the trained LSTM music generation model.
Click "Generate Music" -> the model creates a new sequence, converts it
to MIDI, synthesizes it to audio (no external soundfont needed, uses
pretty_midi's built-in synthesizer), and plays it right in the browser.
Also gives a downloadable .mid file.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud (free):
    See DEPLOY.md in this project for step-by-step instructions.
"""

import pickle
import tempfile
import numpy as np
import streamlit as st
import pretty_midi
from tensorflow import keras
from music21 import stream, note, chord, duration

MODEL_PATH = "model/music_lstm.keras"
MAPPINGS_PATH = "model/mappings.pkl"
NOTES_PATH = "data/notes.pkl"


@st.cache_resource
def load_everything():
    """Load model + data once and cache across reruns/clicks."""
    model = keras.models.load_model(MODEL_PATH)
    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)
    with open(NOTES_PATH, "rb") as f:
        all_events = pickle.load(f)
    return model, mappings, all_events


def sample_with_temperature(probabilities, temperature):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-9) / temperature
    exp_preds = np.exp(probabilities)
    probabilities = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probabilities), p=probabilities)


def generate_notes(model, seed_sequence, int_to_note, vocab_size, length, temperature):
    pattern = list(seed_sequence)
    generated = []
    for _ in range(length):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
        prediction = model.predict(input_seq, verbose=0)[0]
        idx = sample_with_temperature(prediction, temperature)
        generated.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]
    return generated


def events_to_midi_file(events, output_path):
    output_stream = stream.Stream()
    for event in events:
        if ('.' in event) or event.isdigit():
            notes_in_chord = event.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.duration = duration.Duration(0.5)
            output_stream.append(new_chord)
        else:
            new_note = note.Note(event)
            new_note.duration = duration.Duration(0.5)
            output_stream.append(new_note)
    output_stream.write('midi', fp=output_path)


def generate_music(model, mappings, all_events, num_notes, temperature):
    note_to_int = mappings["note_to_int"]
    int_to_note = mappings["int_to_note"]
    vocab_size = mappings["vocab_size"]
    seq_length = mappings["seq_length"]

    start_idx = np.random.randint(0, len(all_events) - seq_length - 1)
    seed_events = all_events[start_idx:start_idx + seq_length]
    seed_sequence = [note_to_int[n] for n in seed_events]

    generated_events = generate_notes(model, seed_sequence, int_to_note, vocab_size, int(num_notes), temperature)

    midi_path = tempfile.NamedTemporaryFile(suffix=".mid", delete=False).name
    events_to_midi_file(generated_events, midi_path)

    pm = pretty_midi.PrettyMIDI(midi_path)
    audio = pm.synthesize(fs=22050)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)  # normalize

    return audio, midi_path


# ---------------- UI ----------------
st.set_page_config(page_title="AI Music Generator", page_icon="🎵")

st.title("🎵 AI Music Generator")
st.markdown(
    """
    An LSTM neural network trained on Bach chorales, generating new original music.
    Click **Generate Music** to create a fresh sequence every time.
    """
)

with st.spinner("Loading model..."):
    model, mappings, all_events = load_everything()

col1, col2 = st.columns(2)
with col1:
    num_notes = st.slider("Number of notes/chords to generate", 50, 400, 200, step=10)
with col2:
    temperature = st.slider("Temperature (creativity)", 0.3, 1.5, 1.0, step=0.1)

if st.button("🎶 Generate Music", type="primary"):
    with st.spinner("Generating..."):
        audio, midi_path = generate_music(model, mappings, all_events, num_notes, temperature)

    st.audio(audio, sample_rate=22050)

    with open(midi_path, "rb") as f:
        st.download_button("⬇️ Download MIDI file", f, file_name="generated_music.mid")

st.markdown(
    """
    ---
    **How it works:** A 2-layer stacked LSTM (256 units) was trained to predict the next
    note/chord given the previous 40, using Bach chorales as training data (via `music21`).
    Generation is autoregressive — each predicted note is fed back in to predict the next one.
    """
)