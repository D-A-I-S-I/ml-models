from grid_search_helper import train_and_evaluate, load_data
from multiprocessing import Pool
from itertools import product
import multiprocessing
import torch
import json

# Hyperparameters
hidden_dims = [8]
embedding_dims = [10]
encoding_dims = [4]
batch_sizes = [128]
learning_rates = [0.01]
sequence_lengths = [5]
patience = 4
num_epochs = 10
val_split = 0.3

# NOTE: Make sure to change paths to location of dataset!
folder_path = '../../../ADFA-LD-Dataset/ADFA-LD/Training_Data_Master/'
attack_data_master_path = '../../../ADFA-LD-Dataset/ADFA-LD/Attack_Data_Master/'

# Create all combinations of hyperparameters

def multiprocessing_wrapper(combination):
    """
    Wrapper function for multiprocessing. Returns the train and attack loss for the given hyperparameters.
    """
    # print("Process ID: {}, Hyperparameters: {}".format(multiprocessing.current_process().pid, combination))
    hidden_dim, embedding_dim, encoding_dim, batch_size, lr, sequence_length, num_epochs, patience, val_split = combination
    train_dataset, train_loader, val_loader, num_unique_syscalls = load_data(folder_path, sequence_length, batch_size, val_split)

    return train_and_evaluate(train_loader, val_loader, sequence_length, num_epochs, train_dataset,
                               attack_data_master_path, num_unique_syscalls, hidden_dim,
                                 embedding_dim, encoding_dim, batch_size, lr, patience)

def run_grid_search(pool, all_combinations):
    """
    Runs the grid search using multiprocessing. Returns the results.
    """
    total = len(all_combinations)
    done = 0
    results = []

    for result in pool.imap_unordered(multiprocessing_wrapper, all_combinations):
        done += 1
        print(f"Progress: {done}/{total} combinations done ({(done/total)*100:.2f}%)")
        results.append(result)

    return results


if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    print('CPU count:', cores)
    
    all_combinations = list(product(hidden_dims, embedding_dims, encoding_dims, batch_sizes, learning_rates, sequence_lengths, [num_epochs], [patience], [val_split]))
    print(f"Total number of combinations: {len(all_combinations)}")
    num_processes = len(all_combinations) if len(all_combinations) < cores else cores

    # Start the grid search WITH multiprocessing
    with Pool(processes=(num_processes)) as pool: # NOTE: Leave 2 cores for other processes, change if desired
        results = run_grid_search(pool, all_combinations)
    
    # Start the grid search WITHOUT multiprocessing
    # results = map(multiprocessing_wrapper, all_combinations)

    # Find the best model
    best_model_info = {
        'train_loss': float('inf'),
        'attack_loss': float('-inf'),
        'val_loss': float('inf'),
        'atk_val_ratio': float('-inf'),
        'hidden_dim': None,
        'embedding_dim': None,
        'encoding_dim': None,
        'batch_size': None,
        'learning_rate': None,
        'sequence_length': None

    }

    # Initialize a list to store the results
    top_models = []

    for result in results:
        train_loss, attack_loss, best_val_loss, best_ratio, hidden_dim, embedding_dim, encoding_dim, batch_size, lr, best_model, sequence_length, total_epochs = result
        if best_ratio > best_model_info['atk_val_ratio']:
            best_model_info.update({
                'train_loss': train_loss,
                'attack_loss': attack_loss,
                'val_loss': best_val_loss,
                'atk_val_ratio': best_ratio,
                'hidden_dim': hidden_dim,
                'embedding_dim': embedding_dim,
                'encoding_dim': encoding_dim,
                'batch_size': batch_size,
                'learning_rate': lr,
                'sequence_length': sequence_length,
                'total_epochs': total_epochs,
                'best_model': best_model
            })
        
        # Add the current model to the list of top models
        top_models.append({
            'train_loss': train_loss,
            'attack_loss': attack_loss,
            'val_loss': best_val_loss,
            'atk_val_ratio': best_ratio,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim,
            'encoding_dim': encoding_dim,
            'batch_size': batch_size,
            'learning_rate': lr,
            'sequence_length': sequence_length,
            'total_epochs': total_epochs,
            'best_model': best_model
        })

    # Sort the top models based on atk_val_ratio in descending order
    top_models.sort(key=lambda x: x['atk_val_ratio'], reverse=True)

    print("Top 10 Models:")
    for i, model in enumerate(top_models[:10]):
        model_info = {k: v for k, v in model.items() if k != 'best_model'}
        print(f"Rank {i+1}: {model_info}")

    # save 5 best models to files, and model info in separate file
    for i, model in enumerate(top_models[:5]):
        torch.save(model['best_model'], f'trained_models/model_{i}.pth')  
        with open(f'trained_models/model_{i}_info.json', 'w') as f:
            model_info = {k: v for k, v in model.items() if k != 'best_model'}
            json.dump(model_info, f)
