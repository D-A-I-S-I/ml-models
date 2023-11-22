from grid_search_helper import train_and_evaluate, load_data, load_all_attack_data
from multiprocessing import Pool
from itertools import product
import multiprocessing

# Hyperparameters
hidden_dims = [16, 32, 64]
embedding_dims = [10, 20, 30]
encoding_dims = [4, 8, 16]
batch_sizes = [256, 512, 1024]
learning_rates = [0.002]
sequence_lengths = [5, 10, 15, 20]
patience = 5
num_epochs = 25
val_split = 0.3

# NOTE: Make sure to change paths accordingly!
folder_path = '../../ADFA-LD-Dataset/ADFA-LD/Training_Data_Master/'
attack_data_master_path = '../../ADFA-LD-Dataset/ADFA-LD/Attack_Data_Master/'

# Create all combinations of hyperparameters

def multiprocessing_wrapper(combination):
    """
    Wrapper function for multiprocessing. Returns the train and attack loss for the given hyperparameters.
    """
    # print("Process ID: {}, Hyperparameters: {}".format(multiprocessing.current_process().pid, combination))
    hidden_dim, embedding_dim, encoding_dim, batch_size, lr, sequence_length, num_epochs, patience, val_split = combination
    train_dataset, train_loader, val_loader, num_unique_syscalls = load_data(folder_path, sequence_length, batch_size, val_split) # TODO: move all load_data calls to helper file

    return train_and_evaluate(train_loader, val_loader, sequence_length, num_epochs, train_dataset,
                               attack_data_master_path, num_unique_syscalls, hidden_dim,
                                 embedding_dim, encoding_dim, batch_size, lr, patience)

def track_progress_and_collect_results(pool, all_combinations):
    total = len(all_combinations)
    done = 0
    results = []  # List to store results

    for result in pool.imap_unordered(multiprocessing_wrapper, all_combinations):
        done += 1
        print(f"Progress: {done}/{total} combinations done ({(done/total)*100:.2f}%)")
        results.append(result)

    return results  # Return the collected results


if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    print('CPU count:', cores)
    
    all_combinations = list(product(hidden_dims, embedding_dims, encoding_dims, batch_sizes, learning_rates, sequence_lengths, [num_epochs], [patience], [val_split]))
    num_processes = len(all_combinations) if len(all_combinations) < cores else cores
    # Start the grid search WITH multiprocessing
    with Pool(processes=num_processes) as pool:
        results = track_progress_and_collect_results(pool, all_combinations)
    
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
        train_loss, attack_loss, best_val_loss, best_ratio, hidden_dim, embedding_dim, encoding_dim, batch_size, lr, best_model, sequence_length = result
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
                'sequence_length': sequence_length
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
            'sequence_length': sequence_length
        })

    # Sort the top models based on atk_val_ratio in descending order
    top_models.sort(key=lambda x: x['atk_val_ratio'], reverse=True)

    # Print the top 10 models
    print("Top 10 Models:")
    for i, model in enumerate(top_models[:10]):
        print(f"Rank {i+1}: {model}")

