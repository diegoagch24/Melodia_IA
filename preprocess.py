import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS = [
    0.25, 
    0.5, 
    0.75,
    1.0, 
    1.5,
    2, 
    3,
    4 
]


def load_songs_in_kern(dataset_path):
    """Loads all kern pieces in dataset using music21.

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    songs = []

    # va a través de todos los archivos en el dataset y cargarlos con music21

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # considera solo los archivos kern
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    """Boolean routine that returns True if piece has all acceptable duration, False otherwise.

    :param song (m21 stream):
    :param acceptable_durations (list): List of acceptable duration in quarter length
    :return (bool):
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """Transposes song to C maj/A min

    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """

    # Obtiene la nota de la canción
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

   # Estima la nota usando music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Obtiene el intervalo para la transposición. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

   # transpone la canción calculando el intervalo
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song, time_step=0.25):
    
    encoded_song = []
    
    for event in song.flat.notesAndRests:
        #controlar las notas
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # invocamos la canción cifrada a una cadena
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song

def preprocess(dataset_path):

    # Cargar canciones folclóricas
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # Filtrar las canciones que tienen duraciones no aceptables
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

         # transponer las canciones a Cmaj/Amin
        song = transpose(song)

        # Codificación de canciones con representación de series temporales de música
        encoded_song = encode_song(song)

        # Guarda las canciones en un archivo de texto
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # Carga las canciones cifradas y añade sus delimitadores
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    # guarda las cadenas que contienen toda la información
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}

    # Identifica el vocabulario
    songs = songs.split()
    vocabulary = list(set(songs))

    # Creación del mappeo
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Guarda el vocabulario en un archivo json
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    

def convert_songs_to_int(songs):

    int_songs = []

    #Carga el mapeo
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    #invoca las canciones encadenadas a una lista
    songs = songs.split()

    #mapeo de las canciones a un entero
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs

def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> i: [11,12], t:13; i: [12,13] , t:14

    # Carga las canciones y hazle un mapeo
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # Genera la secuencia de entrenamiento
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    # codifica las secuencias
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def main ():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)

    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    #a = 1

if __name__ == "__main__":

    main()
    # Canciones cargadas
    #songs = load_songs_in_kern(KERN_DATASET_PATH)
    # print(f"Loaded {len(songs)} songs.")
    #song = songs[0]

    #print(f"Tiene una duración aceptable? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")

    #preprocess(KERN_DATASET_PATH)

    #songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)

    #transposed_song = transpose(song)

    #song.show()
    #transposed_song.show()