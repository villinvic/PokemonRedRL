import os
import time
import zmq

# Replace 'your_named_pipe_path' with the actual path to your named pipe
pipe_path = 'inproc://pokemon_ai'

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

    while True:
        # Generate dummy score
        score = total_exp // episode_id  # Just a placeholder calculation, replace it with your actual scoring logic

        # Format the data
        data = f"{episode_id} {score} {total_exp}"

        # Open the named pipe and write the data

        try:
            with os.fdopen(os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK), 'w') as pipe:
                pipe.write(data)
                pipe.flush()  # Ensure data is flushed to the pipe
                print(f"Sent data: {data}")
        except BlockingIOError:
            print("Pipe is busy. Skipping write.")

        # Print to console for verification (you can comment this line out if not needed)
        print(f"Sent data: {data}")

        # Update episode_id and total_exp for the next iteration
        episode_id += 1
        total_exp += 100  # Just an example increment, replace it with your actual increment logic

        # Sleep for 10 seconds
        time.sleep(5)

if __name__ == "__main__":
    # Ensure the named pipe exists before running the script
    create_named_pipe()
    send_dummy_data()