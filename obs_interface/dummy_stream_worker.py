import os
import time

import numpy as np
import zmq

# Replace 'your_named_pipe_path' with the actual path to your named pipe
pipe_path = 'ipc://pokemon_ai'

def create_named_pipe():
    try:
        os.remove(pipe_path)
    except:
        pass
    os.mkfifo(pipe_path)
    print(f"Named pipe '{pipe_path}' created.")


def send_dummy_data():
    episode_id = 1
    total_exp = 0

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind(pipe_path)
    print("inited")

    to_send = []

    while True:
        # Generate dummy score
        score = total_exp // episode_id  # Just a placeholder calculation, replace it with your actual scoring logic

        # Format the data
        data = {
            "episode_id": episode_id,
            "score": score,
            "total_exp": total_exp,
            "pokemon_ids": list(np.random.choice(152, 6))
        }

        to_send.append(data)

        # Open the named pipe and write the data
        while to_send:
            try:
                socket.send_pyobj(to_send[0], zmq.NOBLOCK)
                to_send.pop(0)
            except Exception:
                break

        # Print to console for verification (you can comment this line out if not needed)
        print(f"Sent data: {data}", f"in queue : {len(to_send)}")

        # Update episode_id and total_exp for the next iteration
        episode_id += 1
        total_exp += 100  # Just an example increment, replace it with your actual increment logic

        # Sleep for 10 seconds
        time.sleep(5)

if __name__ == "__main__":
    # Ensure the named pipe exists before running the script
    # create_named_pipe()
    send_dummy_data()