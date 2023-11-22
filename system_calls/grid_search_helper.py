import torch.optim as optim
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from Autoencoder import Autoencoder
from os import getpid
import time

# ? make into class

def train_and_evaluate(train_loader, val_loader, sequence_length, num_epochs, train_dataset, attack_data_master_path, num_unique_syscalls, hidden_dim, embedding_dim, encoding_dim, batch_size, lr, patience):
    """
    Trains the autoencoder and evaluates it on the attack data. Returns the best model.
    """
    start_time = time.time()

    load_attack_data_start_time = time.time()

    # Cache attack data
    attack_data_cache = load_all_attack_data(attack_data_master_path, sequence_length, batch_size)

    print("Process ID: {}, Attack data loading duration: {:.2f}s".format(getpid(), time.time() - load_attack_data_start_time))

    # Initialize model
    autoencoder = Autoencoder(sequence_length=sequence_length, num_system_calls=num_unique_syscalls, embedding_dim=embedding_dim, encoding_dim=encoding_dim, hidden_dim=hidden_dim)
    optimizer = optim.SGD(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Print which hyperparameters we're using
    print("Training with hidden_dim: {}, embedding_dim: {}, encoding_dim: {}, batch_size: {}, lr: {}".format(hidden_dim, embedding_dim, encoding_dim, batch_size, lr))

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_attack_loss = float('-inf')
    best_ratio = float('-inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Training model
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # print("-- Process ID: {}, Epoch: {}".format(getpid(), epoch))
        autoencoder.train()
        
        # Early stopping
        if epoch > patience and epochs_without_improvement > patience:
            break

        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()

            inputs = batch[0]
            embedding_start_time = time.time()
            embedded_inputs = autoencoder.embedding(inputs)
            embedded_inputs = embedded_inputs.view(inputs.size(0), -1)
            print(f"Embedding time: {time.time() - embedding_start_time:.4f}s")

            forward_start_time = time.time()
            outputs = autoencoder(inputs)
            outputs = outputs.view(inputs.size(0), -1)
            print(f"Forward pass time: {time.time() - forward_start_time:.4f}s")

            loss_start_time = time.time()
            loss = criterion(outputs, embedded_inputs)
            print(f"Loss computation time: {time.time() - loss_start_time:.4f}s")

            backward_start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Backward pass and optimization time: {time.time() - backward_start_time:.4f}s")

            train_loss += loss.item()

            print(f"Total time for batch {i}: {time.time() - batch_start_time:.4f}s\n")
                
        train_loss /= len(train_loader)

        # Validation step
        autoencoder.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                inputs = batch[0]
                embedded_inputs = autoencoder.embedding(inputs).view(inputs.size(0), -1)

                outputs = autoencoder(inputs).view(inputs.size(0), -1)
                loss = criterion(outputs, embedded_inputs)
                val_loss += loss.item()

            val_loss /= len(val_loader)

            # Test on attack data
            attack_test_start = time.time()
            attack_loss = attack_test(autoencoder, criterion, attack_data_cache)
            print("Process ID: {}, Attack test duration: {:.2f}s".format(getpid(), time.time() - attack_test_start))

        # Calculate the relative difference (ratio)
        current_ratio = attack_loss / val_loss if val_loss != 0 else float('inf')

        # Update model saving and early stopping conditions
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if attack_loss > best_attack_loss:
                best_attack_loss = attack_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            epochs_without_improvement = 0
            best_model_state = autoencoder.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            break

        # print("Process ID: {}, Epoch: {}, Train Loss: {}, Attack Loss: {}, Validation Loss: {}, Atk/Val ratio: {}".format(getpid(), epoch, train_loss, attack_loss, val_loss, current_ratio))
        print("Process ID: {}, Epoch duration: {:.2f}s".format(getpid(), time.time() - epoch_start_time))

    print("Total training duration: {:.2f}s".format(time.time() - start_time))
    return best_train_loss, best_attack_loss, best_val_loss, best_ratio, hidden_dim, embedding_dim, encoding_dim, batch_size, lr, best_model_state, sequence_length

# Function to load all attack data
def load_all_attack_data(attack_data_master_path, sequence_length, batch_size):
    attack_data_cache = {}
    for folder_name in os.listdir(attack_data_master_path):
        folder_path = os.path.join(attack_data_master_path, folder_name)
        if os.path.isdir(folder_path):
            attack_data, attack_loader, _, _ = load_data(folder_path, sequence_length, batch_size, val_split=0.0)
            attack_data_cache[folder_name] = (attack_data, attack_loader)
    return attack_data_cache

def attack_test(autoencoder, criterion, attack_data_cache):
    """
    Tests the autoencoder on the attack data. Returns the average attack loss.
    """
    attack_loss = 0.0

    for folder_name, (attack_data, attack_loader) in attack_data_cache.items():
        attack_loss += test_on_folder(autoencoder, criterion, attack_loader)

    return attack_loss/len(attack_data_cache)

def test_on_folder(autoencoder, criterion, attack_loader):
    """
    Tests the autoencoder on the attack data in the given folder. Returns the average loss.
    """
    # Load attack data from the folder
    attack_loss = 0.0

    # Calculate loss for each batch in the attack data
    for batch in attack_loader:
        inputs = batch[0]
        embedded_inputs = autoencoder.embedding(inputs).view(inputs.size(0), -1)
        outputs = autoencoder(inputs).view(inputs.size(0), -1)
        loss = criterion(outputs, embedded_inputs)
        attack_loss += loss.item()

    return attack_loss/len(attack_loader.dataset)

def load_data(folder_path, sequence_length, batch_size, val_split=0.3):
    """
    Loads the data from the given folder path. Returns the dataset and DataLoaders.
    """
    data = []
    unique = []
    # Process each file
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read().split()
                content = [int(s) for s in content]
                combined_batches = [content[i:i + sequence_length] for i in range(0, len(content))]
                combined_batches = [lst for lst in combined_batches if len(lst) == sequence_length]
                data.extend(combined_batches)
                unique.extend(content)

    unique_syscalls = set(unique)
    num_unique_syscalls = 174 # ! temp solution (len(unique_syscalls))
    
    # Create a mapping from system call number to a unique index
    mapping = {sys_call: i for i, sys_call in enumerate(unique_syscalls)}

    # Apply the mapping to each system call in each sequence
    mapped_data = [[mapping[sys_call] for sys_call in sequence] for sequence in data]

    # Create tensors and DataLoader
    tensors = [torch.tensor(x) for x in mapped_data]
    tensor = torch.stack(tensors)
    dataset = TensorDataset(tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return dataset, train_loader, val_loader, num_unique_syscalls