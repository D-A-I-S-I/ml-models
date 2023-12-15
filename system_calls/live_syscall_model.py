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


def live_read_and_process(file_path, model, syscall_mapping, sequence_length, threshold, read_size, batch_size):
    """
    Continuously read and process system call sequences from the given file using a sliding window.
    """
    criterion = nn.MSELoss(reduction='none')
    last_position = 0  # Keep track of the last read position in the file

    # Initialize an empty buffer
    buffer = []

    while True:
        time.sleep(0.5)  # FIXME: adjust or the sleep time later
        print('Waiting for new data...')

        with open(file_path, 'r') as file:
            file.seek(last_position)  # Move to the last read position
            content = file.read(read_size)  # Read 'size' number of bytes
            last_position = file.tell()  # Update the last read position

            syscalls = content.strip().split()
            num_sequences = len(syscalls) - sequence_length + 1 if len(syscalls) >= sequence_length else 0

            start_time = time.time()
            for i in range(num_sequences):
                sequence = syscalls[i:i + sequence_length]
                tensor_sequence = preprocess_sequence(sequence, syscall_mapping, sequence_length)

                # Add the tensor sequence to the buffer
                buffer.append(tensor_sequence)

                # If the buffer is full, process a batch and remove it from the buffer
                if len(buffer) >= batch_size:
                    # Convert the batch to a tensor
                    tensor_batch = torch.stack(buffer[:batch_size])

                    # Extract embedded input
                    embedded_inputs = model.embedding(tensor_batch).view(tensor_batch.size(0), -1)

                    # Get the model's output
                    outputs = model(tensor_batch).view(tensor_batch.size(0), -1)

                    # Compute loss and classify
                    losses = criterion(outputs, embedded_inputs).mean(dim=1)
                    classifications = ['POSSIBLE INTRUSION' if loss.item() > threshold else 'Normal' for loss in losses]

                    # # Print the classifications and losses
                    # for sequence, classification, loss in zip(buffer[:batch_size], classifications, losses):
                    #     formatted_sequence = ', '.join(str(elem) for elem in sequence)
                    #     print(f"Sequence: [{formatted_sequence}] - Classification: {classification}, Loss: {loss.item():.6f}")
                    

                    end_time = time.time()
                    num_sequences = len(buffer[:batch_size])
                    time_taken = end_time - start_time
                    sequences_per_second = num_sequences / time_taken
                    print(f"Processed {num_sequences} sequences in {time_taken:.2f} seconds ({sequences_per_second:.2f} sequences/second)")
                    start_time = time.time()

                    # Remove the processed batch from the buffer
                    buffer = buffer[batch_size:]


if __name__ == '__main__':
    # ~470 syscalls according to https://www.man7.org/linux/man-pages/man2/syscalls.2.html
    # ~310 syscalls according to https://filippo.io/linux-syscall-table/
    # ~380 syscalls according to unist_64.h file

    num_system_calls = 174 # FIXME: change when model is retrained accordingly

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
    threshold = 1.0 # FIXME: adjust after retraining the model

    live_read_and_process(file_path, model, syscall_mapping, sequence_length, threshold, read_size=999999999999, batch_size=10000)