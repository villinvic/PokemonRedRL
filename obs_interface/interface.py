import sys
import os
import time

import zmq
from PyQt5.QtGui import QMovie, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtProperty, QPropertyAnimation, QSequentialAnimationGroup, \
    QEasingCurve


# TODO : animations
# badges
# pokemon info
# history and stats

class FadingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._opacity = 1.0

    def getOpacity(self):
        return self._opacity

    def setOpacity(self, value):
        self._opacity = value
        self.repaint()

    opacity = pyqtProperty(float, getOpacity, setOpacity)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(self.opacity)
        super().paintEvent(event)


class PokemonAIInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        self.pokemon_ids = [0] * 6

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
            pokemon_label = FadingLabel(self)
            layout.addWidget(pokemon_label, i // 3 + 1, i % 3)
            self.pokemon_labels.append(pokemon_label)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Pokemon AI Interface')
        self.setGeometry(100, 100, 500, 300)

    def update_data(self, data):
        # Update episode information labels
        self.episode_label.setText(f'Episode: {data["episode_id"]}')
        self.score_label.setText(f'Score: {data["score"]}')
        self.total_exp_label.setText(f'Total Exp: {data["total_exp"]}')

        # Load and display Pokémon gifs
        for i, pokemon_id in enumerate(data["pokemon_ids"]):

            if pokemon_id == 0 and self.pokemon_ids[i] != 0:
                # pokemon removed from party : TODO
                self.pokemon_labels[i].clear()
            elif pokemon_id != 0 and self.pokemon_ids[i] == 0:
                # pokemon added to party
                self.animate_pokemon_entrance(pokemon_id, self.pokemon_labels[i])
            elif pokemon_id != self.pokemon_ids[i]:
                # changed pokemon at idx i : TODO
                self.load_and_display_gif(pokemon_id, self.pokemon_labels[i])

            self.pokemon_ids[i] = pokemon_id


    def animate_pokemon_entrance(self, new_pokemon_id, label):
        # Load entrance GIF and play the animation
        entrance_gif_path = 'obs_interface/assets/animations/animation7.gif'
        entrance_movie = QMovie(entrance_gif_path)
        label.setMovie(entrance_movie)
        entrance_movie.start()

        # Create a fade-out animation for the entrance GIF
        fade_out_animation = QPropertyAnimation(label, b"opacity")
        fade_out_animation.setEasingCurve(QEasingCurve.InOutCubic)
        fade_out_animation.setStartValue(1.0)
        fade_out_animation.setEndValue(.0)
        fade_out_animation.setDuration(2000)

        # Set up animation sequence
        animation_group = QSequentialAnimationGroup(self)
        animation_group.addAnimation(fade_out_animation)

        # Connect the signal to load and display the actual Pokémon GIF
        animation_group.finished.connect(lambda: self.load_and_display_gif(new_pokemon_id, label))
        animation_group.start()


    def load_and_display_gif(self, pokemon_id, label):
        try:
            # Load the new Pokémon gif
            gif_path = f"obs_interface/assets/sprites/ani_bw_{pokemon_id:03d}.gif"
            movie = QMovie(gif_path)
            label.setMovie(movie)
            movie.start()

        except Exception as e:
            print(f"Error loading gif: {e}")


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