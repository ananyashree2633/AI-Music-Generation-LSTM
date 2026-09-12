"""
Step 1: Data Collection + Preprocessing
----------------------------------------
Uses music21's built-in Bach chorale corpus (433 pieces, public domain,
ships with the library so no download needed). If you want to use your
own MIDI files instead, just point MIDI_FOLDER at a directory of .mid
files and set USE_LOCAL_MIDI = True below.

Output: data/notes.pkl  (a flat list of note/chord "events" as strings)
"""

import pickle
import glob
from music21 import converter, corpus, instrument, note, chord

# ---------------- CONFIG ----------------
USE_LOCAL_MIDI = False          # True -> read .mid files from MIDI_FOLDER instead
MIDI_FOLDER = "data/midi/*.mid"
NUM_PIECES = 60                 # how many pieces to use (more = better model, slower prep)
OUTPUT_PATH = "data/notes.pkl"
# -----------------------------------------


def get_scores():
    """Return a list of music21 Score/Stream objects to train on."""
    if USE_LOCAL_MIDI:
        files = glob.glob(MIDI_FOLDER)
        if not files:
            raise FileNotFoundError(
                f"No .mid files found at {MIDI_FOLDER}. "
                "Either add MIDI files there or set USE_LOCAL_MIDI = False "
                "to use the built-in Bach corpus."
            )
        print(f"Found {len(files)} local MIDI files.")
        return [converter.parse(f) for f in files[:NUM_PIECES]]
    else:
        paths = corpus.getComposer("bach")[:NUM_PIECES]
        print(f"Using {len(paths)} Bach chorales from music21's built-in corpus.")
        return [corpus.parse(p) for p in paths]


def extract_events(score):
    """
    Turn a music21 stream into a flat list of string 'events'.
    - A single note becomes its pitch name, e.g. 'C4'
    - A chord becomes its pitch classes joined by dots, e.g. '4.7.11'
    We flatten all parts together (fine for a first model; you can later
    keep per-voice streams for a more sophisticated polyphonic model).
    """
    events = []
    try:
        parts = instrument.partitionByInstrument(score)
        elements = parts.parts[0].recurse() if parts else score.flat.notes
    except Exception:
        elements = score.flat.notes

    for el in elements:
        if isinstance(el, note.Note):
            events.append(str(el.pitch))
        elif isinstance(el, chord.Chord):
            events.append('.'.join(str(n) for n in el.normalOrder))
    return events


def main():
    import os
    os.makedirs("data", exist_ok=True)

    scores = get_scores()
    all_events = []
    for i, score in enumerate(scores):
        try:
            events = extract_events(score)
            all_events.extend(events)
            print(f"  [{i+1}/{len(scores)}] extracted {len(events)} events")
        except Exception as e:
            print(f"  [{i+1}/{len(scores)}] skipped (parse error: {e})")

    print(f"\nTotal events collected: {len(all_events)}")
    print(f"Unique event vocabulary size: {len(set(all_events))}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_events, f)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()