import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal

# Define the class for the GUI
class PokemonAIInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Create labels for displaying information
        self.episode_label = QLabel('Episode: N/A')
        self.score_label = QLabel('Score: N/A')
        self.total_exp_label = QLabel('Total exp: N/A')


        # Add labels to the layout
        layout.addWidget(self.episode_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.total_exp_label)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Pokemon AI Interface')
        self.setGeometry(100, 100, 300, 200)

    def update_data(self, episode, score, total_exp):
        # Update the labels with new data
        self.episode_label.setText(f'Episode: {episode}')
        self.score_label.setText(f'Score: {score}')
        self.total_exp_label.setText(f'Total exp: {total_exp}')


# Define a thread to read data from the named pipe
class NamedPipeReader(QThread):
    update_signal = pyqtSignal(str, str, str)

    def __init__(self, pipe_path):
        super().__init__()
        self.pipe_path = pipe_path

    def run(self):
        with open(self.pipe_path, 'r') as pipe:
            while True:
                data = pipe.readline().strip()
                if not data:
                    time.sleep(1.)  # Add a short delay to avoid high CPU usage
                    continue

                # Assuming the data format is "episode_info stats_info"
                episode_id, score, total_exp = data.split()
                self.update_signal.emit(episode_id, score, total_exp)


if __name__ == '__main__':
    # Set up the named pipe path
    pipe_path = 'inproc://pokemon_ai'

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