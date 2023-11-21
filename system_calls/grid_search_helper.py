import torch.optim as optim
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Autoencoder import Autoencoder

# ? make into class

def train_and_evaluate(train_loader, sequence_length, num_epochs, train_dataset, attack_data_master_path, num_unique_syscalls, hidden_dim, embedding_dim, encoding_dim, batch_size, lr):
    """
    Trains the autoencoder and evaluates it on the attack data. Returns the best model.
    """
    # Initialize model
    autoencoder = Autoencoder(sequence_length=sequence_length, num_system_calls=num_unique_syscalls, embedding_dim=embedding_dim, encoding_dim=encoding_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Print which hyperparameters we're using
    print("Training with hidden_dim: {}, embedding_dim: {}, encoding_dim: {}, batch_size: {}, lr: {}".format(hidden_dim, embedding_dim, encoding_dim, batch_size, lr))

    # DataLoaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_train_loss = float('inf')
    best_attack_loss = float('-inf')

    for epoch in range(num_epochs):
        # Training
        # autoencoder.train()
        
        # TODO: Early stopping
        # TODO: Save best model

        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0]
            embedded_inputs = autoencoder.embedding(inputs)
            embedded_inputs = embedded_inputs.view(inputs.size(0), -1)

            # Forward pass and loss computation
            outputs = autoencoder(inputs)
            outputs= outputs.view(inputs.size(0), -1)
            
            loss = criterion(outputs, embedded_inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Test on attack data
        attack_loss = attack_test(autoencoder, criterion, attack_data_master_path, sequence_length, batch_size)

        if train_loss < best_train_loss and attack_loss > best_attack_loss:
            best_train_loss = train_loss
            best_attack_loss = attack_loss

        print("Epoch: {}, Train Loss: {}, Attack Loss: {}".format(epoch, train_loss, attack_loss))

    return best_train_loss, best_attack_loss, hidden_dim, embedding_dim, encoding_dim, batch_size, lr


def attack_test(autoencoder, criterion, attack_data_master_path, sequence_length, batch_size):
    """
    Tests the autoencoder on the attack data. Returns the average attack loss.
    """
    attack_losses = []

    # Process each attack data folder
    for folder_name in os.listdir(attack_data_master_path):
        folder_path = os.path.join(attack_data_master_path, folder_name)
        if os.path.isdir(folder_path):
            attack_loss = test_on_folder(autoencoder, criterion, folder_path, sequence_length, batch_size )
            attack_losses.append(attack_loss)

    # Calculate and print average attack loss
    avg_attack_loss = np.mean(attack_losses)

    return avg_attack_loss

def test_on_folder(autoencoder, criterion, folder_path, sequence_length, batch_size):
    """
    Tests the autoencoder on the attack data in the given folder. Returns the average attack loss.
    """
    # Load attack data from the folder
    attack_data, attack_loader, num_unique_syscalls = load_data(folder_path, sequence_length, batch_size)
    attack_loss = 0.0

    # Calculate loss for each batch in the attack data
    for batch in attack_loader:
        inputs = batch[0]
        embedded_inputs = autoencoder.embedding(inputs).view(inputs.size(0), -1)
        outputs = autoencoder(inputs).view(inputs.size(0), -1)
        loss = criterion(outputs, embedded_inputs)
        attack_loss += loss.item()

    return attack_loss/len(attack_data)

def load_data(folder_path, sequence_length, batch_size):
    """
    Loads the data from the given folder path. Returns the dataset and DataLoader.
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataset, dataloader, num_unique_syscalls