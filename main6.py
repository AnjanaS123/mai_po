import streamlit as st
import tensorflow as tf
import magenta.music as mm
import magenta.models.music_vae as mv
import numpy as np
import pretty_midi

# Define function to generate music
def generate_music(mood):
    # Define the path to the checkpoint file of the pre-trained MusicVAE model
    checkpoint_file = 'C:/Users/Anjana/Downloads/checkpoint'

    # Load the pre-trained MusicVAE model and get the encoder and decoder objects
    model = mv.TrainedModel(checkpoint_file, mv.Config())
    encoder = model.encoder
    decoder = model.decoder

    # Create a new data converter and assign it to the model config
    data_converter = mm.MidiToNoteSequenceConverter()
    model_config = model.config
    model_config.data_converter = data_converter
    model_config.data_converter.max_tensors_per_notesequence = None

    # Convert mood input to a one-hot encoding vector
    mood_vectors = {'happy': np.array([1, 0, 0]), 'sad': np.array([0, 1, 0]), 'calm': np.array([0, 0, 1])}
    mood_vector = mood_vectors[mood]

    # Generate music using the pre-trained model
    generated_music = model.sample(n=1, length=80, condition=mood_vector)

    # Convert music to MIDI format
    midi_data = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    instrument = pretty_midi.Instrument(program=instrument_program)
    for note in generated_music[0]:
        note_pitch = int(round(note[0] * 127))
        note_velocity = int(round(note[1] * 127))
        note_start = note[2] * 0.5
        note_end = note[3] * 0.5
        midi_note = pretty_midi.Note(
            velocity=note_velocity, pitch=note_pitch, start=note_start, end=note_end)
        instrument.notes.append(midi_note)
    midi_data.instruments.append(instrument)

    # Save generated music as a MIDI file
    midi_data.write('generated_music.mid')

    # Convert MIDI to NoteSequence
    sequence = mm.midi_file_to_note_sequence('generated_music.mid')

    # Quantize the NoteSequence to a fixed grid
    qns = mm.sequences_lib.quantize_note_sequence(sequence, steps_per_quarter=4)

    # Convert quantized NoteSequence back to MIDI file
    midi_data = mm.midi_file_to_note_sequence(qns)

    # Save generated music as a MIDI file
    midi_data.write('generated_music.mid')

    # Return the path to the generated music file
    return 'generated_music.mid'

# Define Streamlit app layout
st.title('Music Generation App')
mood = st.selectbox('Select a mood:', ['happy', 'sad', 'calm'])
if st.button('Generate Music'):
    generated_music_path = generate_music(mood)
    st.audio(generated_music_path)
