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
st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="centered")

# --- Custom styling ---
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.6rem;
        background: linear-gradient(90deg, #a855f7, #ec4899, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #b0b0c0;
        font-size: 1.05rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #a855f7, #ec4899);
        border: none;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        border: none;
    }
    .info-card {
        background-color: #1a1a2e;
        border: 1px solid #33334d;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .stats-row {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin: 1rem 0;
    }
    .stat-box {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        flex: 1;
        margin: 0 0.3rem;
    }
    .stat-box h3 {
        margin: 0;
        color: #a855f7;
        font-size: 1.4rem;
    }
    .stat-box p {
        margin: 0;
        color: #9090a5;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🎵 AI Music Generator</h1>
        <p>A neural network trained on Bach chorales — composing original music, one note at a time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("🎼 Warming up the neural network..."):
    model, mappings, all_events = load_everything()

st.markdown(
    f"""
    <div class="stats-row">
        <div class="stat-box"><h3>{mappings['vocab_size']}</h3><p>Unique notes/chords</p></div>
        <div class="stat-box"><h3>{mappings['seq_length']}</h3><p>Context window</p></div>
        <div class="stat-box"><h3>LSTM</h3><p>Architecture</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col1, col2 = st.columns(2)
with col1:
    num_notes = st.slider("🎹 Number of notes/chords", 50, 400, 200, step=10)
with col2:
    temperature = st.slider("🔥 Creativity (temperature)", 0.3, 1.5, 1.0, step=0.1)

st.write("")
generate_clicked = st.button("🎶 Generate Music")

if generate_clicked:
    with st.spinner("Composing your track..."):
        audio, midi_path = generate_music(model, mappings, all_events, num_notes, temperature)

    st.success("Your music is ready! 🎉")
    st.audio(audio, sample_rate=22050)

    with open(midi_path, "rb") as f:
        st.download_button("⬇️ Download MIDI file", f, file_name="generated_music.mid")

    st.balloons()

st.markdown(
    """
    <div class="info-card">
    <b>🧠 How it works</b><br>
    A 2-layer stacked LSTM (256 units each) was trained to predict the next note/chord
    given the previous 40, using Bach chorales as training data (via <code>music21</code>).
    Generation is <b>autoregressive</b> — each predicted note is fed back into the model
    to predict the next one, and the <b>temperature</b> slider controls how bold vs.
    predictable the sampling is.
    </div>
    """,
    unsafe_allow_html=True,
)