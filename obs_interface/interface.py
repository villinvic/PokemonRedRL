import sys
import os
import time

import zmq
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QMovie, QPainter, QFont, QFontDatabase, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout, QGraphicsOpacityEffect, QLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtProperty, QPropertyAnimation, QSequentialAnimationGroup, \
    QEasingCurve, QParallelAnimationGroup, QSize, QRect

# TODO : animations
# badges
# pokemon info
# history and stats
# -> Best so far
# How long have we been learning

# IDEAS
"""
- Cool timer
- Show beautiful graph of learing curve ?
- Caught pokemons ?

- Simple stats with RED/GREEN arrows showing tendency:
    - Maximum - avg - min score
    - Max exp 
    - Max badges ...
"""

class PokemonAIInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        self.pokemon_ids = [0] * 6

    def init_ui(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 177, 64))
        self.setPalette(palette)

        self.setFixedSize(1920, 1080)
        # Set up the layout

        self.game_placeholder = QLabel(self)
        #self.game_placeholder.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.game_placeholder.move(14, 14)
        pixmap = QPixmap("obs_interface/assets/test/game_sample.jpeg")
        pixmap.setDevicePixelRatio(0.2)
        self.game_placeholder.setPixmap(pixmap)
        self.game_placeholder.show()

        self.trainer_card_label = QLabel(self)
        #self.trainer_card_label.setFixedSize(600, 800)

        pixmap = QPixmap("obs_interface/assets/ui/trainer_card/trainer_card_0_badges.png")
        pixmap.setDevicePixelRatio(0.5)
        self.trainer_card_label.setPixmap(pixmap)
        self.trainer_card_label.show()
        self.trainer_card_label.move(800, 400)

        # self.obs_layout = QLabel(self)
        # pixmap = QPixmap("obs_interface/assets/ui/layout/rby_deepred_160x144.png")
        # pixmap.setDevicePixelRatio(1.9)
        # self.obs_layout.setPixmap(pixmap)
        # self.obs_layout.show()
        # #self.trainer_card_label.move(800, 400)



        # Create labels for episode information
        self.episode_label = QLabel('Episode: N/A', self)
        self.episode_label.setAlignment(Qt.AlignLeft)
        #self.episode_label.setFont(QFont)
        self.episode_label.setFixedSize(200, 24)
        self.episode_label.move(10, 10)
        self.score_label = QLabel('Score: N/A', self)
        self.score_label.setFixedSize(200, 24)
        self.score_label.setAlignment(Qt.AlignLeft)


        self.score_label.move(110, 10)
        self.total_exp_label = QLabel('Total Exp: N/A', self)
        self.total_exp_label.setFixedSize(200, 24)
        self.total_exp_label.move(210, 10)
        self.total_exp_label.setAlignment(Qt.AlignLeft)


        # Add episode information labels to the layout

        # Create labels for Pokémon gifs
        self.pokemon_labels = []
        self.pokemon_animation_labels = []
        self.pokemon_stat_labels = []


        for i in range(6):
            pokemon_label = QLabel(self)
            pokemon_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            pokemon_label.setFixedSize(400, 400)

            pokemon_animation_label = QLabel(self)
            pokemon_animation_label.setAlignment(Qt.AlignCenter)
            pokemon_animation_label.setFixedSize(400, 400)

            pokemon_stats_label = QLabel("", self)
            pokemon_stats_label.setAlignment(Qt.AlignCenter)
            pokemon_stats_label.setFixedSize(100, 24)

            x = -6 + i * 50
            y = 100 + 14 + 128
            pokemon_label.move(x, y)
            pokemon_animation_label.move(x, y + 40)
            pokemon_stats_label.move(x, y + 110)


            #layout.addWidget(pokemon_label, i // 3 + 1, i % 3)
            #layout.addWidget(pokemon_animation_label, i // 3 + 1, i % 3)

            self.pokemon_labels.append(pokemon_label)
            self.pokemon_animation_labels.append(pokemon_animation_label)
            self.pokemon_stat_labels.append(pokemon_stats_label)


        # Set the layout for the main window

        # Set window properties
        self.setWindowTitle('Pokemon AI Interface')
        self.setGeometry(100, 100, 1920, 1080)

    def update_data(self, data):
        # Update episode information labels

        self.episode_label.setText(f'Episode: {data["episode_id"]}')
        self.score_label.setText(f'Score: {data["score"]}')
        self.total_exp_label.setText(f'Total Exp: {data["total_exp"]}')

        # Load and display Pokémon gifs
        for i, pokemon_id in enumerate(data["pokemon_ids"]):

            if pokemon_id != 0:
                self.pokemon_stat_labels[i].setText(f"Lv {data['pokemon_levels'][i]}")

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
        curr_size = pokemon_movie.scaledSize()
        pokemon_movie.setScaledSize(QSize(curr_size.width()*8, curr_size.height()*8))

        pokemon_label.setMovie(pokemon_movie)
        pokemon_movie.start()

        entrance_gif_path = 'obs_interface/assets/animations/animation7.gif'
        entrance_movie = QMovie(entrance_gif_path)
        entrance_movie.setSpeed(200)
        animation_label.setMovie(entrance_movie)
        entrance_movie.start()

        # Create a fade-out animation for the entrance GIF
        animation_effect = QGraphicsOpacityEffect()
        animation_label.setGraphicsEffect(animation_effect)
        wait_animation = QPropertyAnimation(animation_effect, b"opacity")
        wait_animation.setStartValue(1.0)
        wait_animation.setEndValue(1.0)
        wait_animation.setDuration(600)
        fade_out_animation = QPropertyAnimation(animation_effect, b"opacity")
        fade_out_animation.setStartValue(1.0)
        fade_out_animation.setEndValue(0.0)
        #fade_out_animation.setEasingCurve(QEasingCurve.InCubic)
        fade_out_animation.setDuration(200)
        pokeball_thrown_animation_full_animation = QSequentialAnimationGroup(self)
        pokeball_thrown_animation_full_animation.addAnimation(wait_animation)
        pokeball_thrown_animation_full_animation.addAnimation(fade_out_animation)

        # Create a fade-in animation for the Pokémon appearance
        pokemon_effect = QGraphicsOpacityEffect()
        pokemon_label.setGraphicsEffect(pokemon_effect)
        wait_animation = QPropertyAnimation(pokemon_effect, b"opacity")
        wait_animation.setStartValue(0.0)
        wait_animation.setEndValue(0.0)
        wait_animation.setDuration(600)

        fade_in_animation = QPropertyAnimation(pokemon_effect, b"opacity")
        fade_in_animation.setStartValue(0.0)
        fade_in_animation.setEndValue(1.0)
        fade_in_animation.setDuration(200)

        # Wont work because widget has fixed size

        spawn_pokemon_full_animation = QSequentialAnimationGroup(self)
        spawn_pokemon_full_animation.addAnimation(wait_animation)
        spawn_pokemon_full_animation.addAnimation(fade_in_animation)

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

    #print(QFontDatabase().families())

    app = QApplication(sys.argv)

    _id = QtGui.QFontDatabase.addApplicationFont("obs_interface/assets/fonts/RBY.ttf")
    custom_font = QFont("PKMN RBYGSC", 16, 8, False)
    app.setFont(custom_font, "QLabel")

    # Create the GUI window
    window = PokemonAIInterface()
    window.show()

    # Create and start the named pipe reader thread
    pipe_reader = NamedPipeReader(pipe_path)
    pipe_reader.update_signal.connect(window.update_data)
    pipe_reader.start()

    # Run the application
    sys.exit(app.exec_())