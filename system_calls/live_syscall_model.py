import torch
import json
import time

from Autoencoder import Autoencoder
from torch import nn


def model_init(model_info: str, num_system_calls: int = 500):
    """
    Initialize the model and load the model info.

    Args:
        model_info (str): path to the model info file
        num_system_calls (int): number of system calls used in training, default to 500
    """

    # Load and create model
    model_state = torch.load('trained_models/model_0.pth')
    model = Autoencoder(model_info['sequence_length'], num_system_calls, model_info['embedding_dim'], model_info['encoding_dim'], model_info['hidden_dim']) 
    model.load_state_dict(model_state)
    model.eval()
    return model

def preprocess_sequence(sequence, mapping, sequence_length):
    """
    Preprocess a single sequence of system calls using the provided mapping.
    """
    # Map the system calls to indices and pad/truncate the sequence
    mapped_sequence = [mapping.get(int(call), 0) for call in sequence]
    if len(mapped_sequence) < sequence_length:
        mapped_sequence += [0] * (sequence_length - len(mapped_sequence))
    else:
        mapped_sequence = mapped_sequence[:sequence_length]

    return torch.tensor(mapped_sequence, dtype=torch.long).unsqueeze(0)  # Use torch.long



def live_read_and_process(file_path, model, syscall_mapping, sequence_length, threshold):
    """
    Continuously read and process system call sequences from the given file using a sliding window.
    """
    criterion = nn.MSELoss()
    last_position = 0  # Keep track of the last read position in the file

    while True:
        time.sleep(0.5)  # FIXME: adjust or the sleep time later

        # TODO: Only read specified number of bytes each time
        # - Finetune based on performance and time to read+process

        # Time logging
        start_read_time = time.time()
        with open(file_path, 'r') as file:
            file.seek(last_position)  # Move to the last read position
            content = file.read()
            last_position = file.tell()  # Update the last read position

            end_read_time = time.time()
            print(f"Read {len(content)} bytes in {end_read_time - start_read_time} seconds") if content else print("Waiting for input data...")

            start_process_time = time.time()

            syscalls = content.strip().split()
            num_sequences = len(syscalls) - sequence_length + 1 if len(syscalls) >= sequence_length else 0
            print(f"Number of sequences: {num_sequences}")
            # Process the content in a sliding window (sequence) manner
            for i in range(num_sequences):
                sequence = syscalls[i:i + sequence_length]
                formatted_sequence = ', '.join(f"{elem:>3}" for elem in sequence) # printing formatting
                tensor_sequence = preprocess_sequence(sequence, syscall_mapping, sequence_length)

                # Extract embedded input
                embedded_inputs = model.embedding(tensor_sequence).view(tensor_sequence.size(0), -1)

                # Get the model's output
                outputs = model(tensor_sequence).view(tensor_sequence.size(0), -1)

                # Compute loss and classify
                loss = criterion(outputs, embedded_inputs)
                classification = 'POSSIBLE INTRUSION' if loss.item() > threshold else 'Normal'
                print(f"Sequence: [{formatted_sequence}] - Classification: {classification}, Loss: {loss.item():.6f}")

            end_process_time = time.time()
            print(f"Processed {num_sequences} sequences in {end_process_time - start_process_time} seconds")



if __name__ == '__main__':
    # 470 system calls according to https://www.man7.org/linux/man-pages/man2/syscalls.2.html
    # (round to 500 to be safe)
    # num_system_calls = 500
    num_system_calls = 174 # FIXME: change to 500 when model is retrained accordingly

    # Load the model info
    with open('trained_models/model_0_info.json', 'r') as f:
        model_info = json.load(f)

    model = model_init(model_info, num_system_calls)

    # Map system call IDs to consecutive integer range [0, num_system_calls)
    syscall_mapping = {syscall: i for i, syscall in enumerate(range(num_system_calls))}

    # Continuous processing
    # TODO: abstract file path to a config file and load continuously in the read loop.
    file_path = 'live_syscall_test_input.txt'
    sequence_length = model_info['sequence_length']
    
    # TODO: abstract the threshold to a config file and load continuously in the read loop.
    threshold = 0.85 # FIXME: adjust after retraining the model

    live_read_and_process(file_path, model, syscall_mapping, sequence_length, threshold)
