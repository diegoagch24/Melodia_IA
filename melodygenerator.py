import json
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21
import os

class MelodyGenerator:
    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)


        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        #Crea la semilla empiece con simbolos

        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed


        # mapea la semilla a un entero

        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # Limita la semilla a max_sequence_length
            seed = seed[-max_sequence_length:]

            # cifra la semilla

            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            onehot_seed = onehot_seed[np.newaxis, ...]

            # haz una predicción

            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)

            # actualizar la semilla
            seed.append(output_int)

            # mapear el entero de nuestro cifrado

            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # verificar la temperatura si estamos al final de la melodia

            if output_symbol == "/":
                break
            # Actualiza la melodia
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilities, temperature):

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25,format="midi", file_name="mel.mid"):

    # crea un flujo de music21
        stream = m21.stream.Stream()

    # analizar todos los símbolos de la melodía y crear objetos nota/resto
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter  # esto equivale a una nota: 0.25 * 4 = 1

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    #reinicia el paso
                    step_counter = 1

                start_symbol = symbol
    # escribe el flujo de m21 en un archivo midi
            else:
                step_counter += 1

        stream.write(format, file_name)
if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ _ _ 67 _ _ _ 64 _ _ _ 60 _"
    seed2 = "55 _ 60 _ _ _ 60 _ 64 _ 62 _ 60 _"
    melody = mg.generate_melody(seed, 500,SEQUENCE_LENGTH, 0.3 )
    print(melody)
    mg.save_melody(melody)
