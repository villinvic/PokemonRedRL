import sys
import os
import time

import zmq
from PyQt5.QtGui import QMovie, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout, QGraphicsOpacityEffect
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtProperty, QPropertyAnimation, QSequentialAnimationGroup, \
    QEasingCurve, QParallelAnimationGroup, QSize, QRect

# TODO : animations
# badges
# pokemon info
# history and stats

def fade(self, widget):
    self.effect = QGraphicsOpacityEffect()
    widget.setGraphicsEffect(self.effect)

    self.animation = QPropertyAnimation(self.effect, b"opacity")
    self.animation.setDuration(1000)
    self.animation.setStartValue(1)
    self.animation.setEndValue(0)
    self.animation.start()

def unfade(self, widget):
    self.effect = QGraphicsOpacityEffect()
    widget.setGraphicsEffect(self.effect)

    self.animation = QPropertyAnimation(self.effect, b"opacity")
    self.animation.setDuration(1000)
    self.animation.setStartValue(0)
    self.animation.setEndValue(1)
    self.animation.start()


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
        self.pokemon_animation_labels = []

        for i in range(6):
            pokemon_label = QLabel(self)
            pokemon_label.setAlignment(Qt.AlignCenter)
            pokemon_label.setFixedSize(100, 100)
            pokemon_animation_label = QLabel(self)
            pokemon_animation_label.widget
            pokemon_animation_label.setAlignment(Qt.AlignCenter)
            pokemon_animation_label.setFixedSize(100, 100)

            layout.addWidget(pokemon_label, i // 3 + 1, i % 3)
            layout.addWidget(pokemon_animation_label, i // 3 + 1, i % 3)

            self.pokemon_labels.append(pokemon_label)
            self.pokemon_animation_labels.append(pokemon_animation_label)


        # Set the layout for the main window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Pokemon AI Interface')
        self.setGeometry(100, 100, 200, 200)

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
                self.animate_pokemon_entrance(pokemon_id, i)
            elif pokemon_id != self.pokemon_ids[i]:
                # changed pokemon at idx i : TODO
                self.load_and_display_gif(pokemon_id, self.pokemon_labels[i])

            self.pokemon_ids[i] = pokemon_id


    def animate_pokemon_entrance(self, new_pokemon_id, idx):

        pokemon_label = self.pokemon_labels[idx]
        animation_label = self.pokemon_animation_labels[idx]

        # Load entrance GIF and play the animation
        pokemon_gif_path = f"obs_interface/assets/sprites/ani_bw_{new_pokemon_id:03d}.gif"
        pokemon_movie = QMovie(pokemon_gif_path)
        pokemon_label.setMovie(pokemon_movie)
        pokemon_movie.start()

        entrance_gif_path = 'obs_interface/assets/animations/animation7.gif'
        entrance_movie = QMovie(entrance_gif_path)
        entrance_movie.setSpeed(150)
        animation_label.setMovie(entrance_movie)
        entrance_movie.start()

        # Create a fade-out animation for the entrance GIF
        animation_effect = QGraphicsOpacityEffect()
        animation_label.setGraphicsEffect(animation_effect)
        wait_animation = QPropertyAnimation(animation_effect, b"opacity")
        wait_animation.setStartValue(1.0)
        wait_animation.setEndValue(1.0)
        wait_animation.setDuration(800)
        fade_out_animation = QPropertyAnimation(animation_effect, b"opacity")
        fade_out_animation.setStartValue(1.0)
        fade_out_animation.setEndValue(0.0)
        #fade_out_animation.setEasingCurve(QEasingCurve.InCubic)
        fade_out_animation.setDuration(400)
        pokeball_thrown_animation_full_animation = QSequentialAnimationGroup(self)
        pokeball_thrown_animation_full_animation.addAnimation(wait_animation)
        pokeball_thrown_animation_full_animation.addAnimation(fade_out_animation)

        # Create a fade-in animation for the Pokémon appearance
        pokemon_effect = QGraphicsOpacityEffect()
        pokemon_label.setGraphicsEffect(pokemon_effect)
        wait_animation = QPropertyAnimation(pokemon_effect, b"opacity")
        wait_animation.setStartValue(0.0)
        wait_animation.setEndValue(0.0)
        wait_animation.setDuration(900)

        fade_in_animation = QPropertyAnimation(pokemon_effect, b"opacity")
        fade_in_animation.setStartValue(0.0)
        fade_in_animation.setEndValue(1.0)
        fade_in_animation.setDuration(200)

        # Wont work because widget has fixed size
        small_to_normal_animation = QPropertyAnimation(pokemon_movie, b"size")

        small_to_normal_animation.setStartValue(QSize(0 , 0))
        small_to_normal_animation.setEndValue(pokemon_label.geometry().size())
        small_to_normal_animation.setDuration(1500)

        spawn_pokemon_animation = QParallelAnimationGroup(self)
        spawn_pokemon_animation.addAnimation(small_to_normal_animation)
        spawn_pokemon_animation.addAnimation(fade_in_animation)

        spawn_pokemon_full_animation = QSequentialAnimationGroup(self)
        spawn_pokemon_full_animation.addAnimation(wait_animation)
        spawn_pokemon_full_animation.addAnimation(spawn_pokemon_animation)

        # Set up animation groups for concurrent execution
        parallel_group = QParallelAnimationGroup(self)
        parallel_group.addAnimation(pokeball_thrown_animation_full_animation)
        parallel_group.addAnimation(spawn_pokemon_full_animation)

        parallel_group.start()


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