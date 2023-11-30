import sys
import os
import time

import zmq
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal

# TODO : animations
# badges
# pokemon info
# history and stats

class PokemonAIInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Set up the layout
        layout = QGridLayout()

        # Create labels for episode information
        self.episode_label = QLabel('Episode: N/A')
        self.score_label = QLabel('Score: N/A')
        self.total_exp_label = QLabel('Total Exp: N/A')

        # Add episode information labels to the layout
        layout.addWidget(self.episode_label, 0, 0)
        layout.addWidget(self.score_label, 0, 1)
        layout.addWidget(self.total_exp_label, 0, 2)

        # Create labels for Pokémon gifs
        self.pokemon_labels = []

        for i in range(6):
            pokemon_label = QLabel(self)
            layout.addWidget(pokemon_label, i // 3 + 1, i % 3)
            self.pokemon_labels.append(pokemon_label)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Pokemon AI Interface')
        self.setGeometry(100, 100, 600, 400)

    def update_data(self, data):
        # Update episode information labels
        self.episode_label.setText(f'Episode: {data["episode_id"]}')
        self.score_label.setText(f'Score: {data["score"]}')
        self.total_exp_label.setText(f'Total Exp: {data["total_exp"]}')

        # Load and display Pokémon gifs
        for i, pokemon_id in enumerate(data["pokemon_ids"]):
            self.load_and_display_gif(pokemon_id, self.pokemon_labels[i])

    def load_and_display_gif(self, pokemon_id, label: QLabel):
        if pokemon_id not in (0, 1,2,3,7,5,6,8,9, 255):
            try:
                # For GIFs
                movie = QMovie(f"obs_interface/assets/sprites/ani_bw_{pokemon_id:03d}.gif")
                label.setMovie(movie)
                movie.start()

            except Exception as e:
                print(f"Error loading gif: {e}")
        else:
            label.clear()


# Define a thread to read data from the named pipe
class NamedPipeReader(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self, pipe_path):
        super().__init__()
        self.pipe_path = pipe_path

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(self.pipe_path)

        while True:
            data = socket.recv_pyobj()
            self.update_signal.emit(data)

if __name__ == '__main__':
    # Set up the named pipe path
    pipe_path = 'ipc://pokemon_ai'

    # Start the PyQt5 application
    app = QApplication(sys.argv)

    # Create the GUI window
    window = PokemonAIInterface()
    window.show()

    # Create and start the named pipe reader thread
    pipe_reader = NamedPipeReader(pipe_path)
    pipe_reader.update_signal.connect(window.update_data)
    pipe_reader.start()

    # Run the application
    sys.exit(app.exec_())