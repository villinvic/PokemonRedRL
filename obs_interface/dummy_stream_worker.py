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
    time_left = 0
    total_exp = 0
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind(pipe_path)
    to_send = []

    while True:
        # Generate dummy score

        if time_left == 0:
            episode_id += 1
            time_left = 3600
            badges = 0
            money = 2000
            pokemons = [107, 0, 0, 0, 0, 0]
            pokemon_levels = [6, 0, 0, 0, 0, 0]
            pokemon_max_healths = [32, 0, 0, 0, 0, 0]
            pokemon_current_healths = [32, 0, 0, 0, 0, 0]
            num_pokemons = 1


        score = total_exp // episode_id  # Just a placeholder calculation, replace it with your actual scoring logic

        if np.random.random() < 0.02 and num_pokemons < 6:
            pokemons[num_pokemons] = np.random.choice(151) + 1
            num_pokemons += 1
        if np.random.random() < 0.1:
            money += np.random.randint(-1000, 1000)
            money = np.clip(money, 0, 100_000)
        if np.random.random()<0.05 and badges < 8:
            badges += 1

        for i in range(6):
            if pokemons[i] != 0:
                if pokemon_levels[i] == 0:
                    pokemon_levels[i] = np.random.randint(1, 50)

                if pokemon_max_healths[i] == 0:
                    pokemon_max_healths[i] = np.random.randint(2, 7) * pokemon_levels[i]
                    pokemon_current_healths[i] = pokemon_max_healths[i]

                if np.random.random() < 0.15 and pokemon_levels[i] < 100:
                    pokemon_max_healths[i] = int(pokemon_max_healths[i] * (pokemon_levels[i] + 1) / pokemon_levels[i])

                    pokemon_levels[i] += 1

                if np.random.random() < 0.15:
                    if np.random.random() < 0.05:
                        pokemon_current_healths[i] = pokemon_max_healths[i]
                    else:
                        pokemon_current_healths[i] -= np.random.randint(32)
                        if pokemon_current_healths[i] < 0:
                            pokemon_current_healths[i] = 0



        # Format the data
        data = {
            "episode_id": episode_id,
            "score": score,
            "total_exp": total_exp,
            "pokemon_ids": pokemons,
            "pokemon_levels": pokemon_levels,
            "pokemon_max_healths": pokemon_max_healths,
            "pokemon_current_healths": pokemon_current_healths,

            "badges": badges,
            "money": money,
            "time_left": time_left,
            # exp
            # badges
            # see environment actually
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

        # Sleep for 10 seconds
        time.sleep(3)
        time_left -= 1

if __name__ == "__main__":
    # Ensure the named pipe exists before running the script
    # create_named_pipe()
    send_dummy_data()