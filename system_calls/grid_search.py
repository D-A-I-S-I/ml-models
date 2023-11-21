from grid_search_helper import train_and_evaluate, load_data
from multiprocessing import Pool
from itertools import product
import multiprocessing

# Hyperparameters
hidden_dims = [16, 32, 64]
embedding_dims = [10, 20, 30]
encoding_dims = [4, 8, 16]
batch_sizes = [256, 512, 1024]
learning_rates = [0.001]
sequence_lengths = [7, 14]
num_epochs = 6

# NOTE: Make sure to change paths accordingly!
folder_path = '../../ADFA-LD-Dataset/ADFA-LD/Training_Data_Master/'
attack_data_master_path = '../../ADFA-LD-Dataset/ADFA-LD/Attack_Data_Master/'

# Create all combinations of hyperparameters
all_combinations = list(product(hidden_dims, embedding_dims, encoding_dims, batch_sizes, learning_rates, sequence_lengths, [num_epochs]))

def multiprocessing_wrapper(combination):
    """
    Wrapper function for multiprocessing. Returns the train and attack loss for the given hyperparameters.
    """
    hidden_dim, embedding_dim, encoding_dim, batch_size, lr, sequence_length, num_epochs = combination
    train_dataset, train_loader, num_unique_syscalls = load_data(folder_path, sequence_length, batch_size) # TODO: move all load_data calls to helper file

    return train_and_evaluate(train_loader, sequence_length, num_epochs, train_dataset,
                               attack_data_master_path, num_unique_syscalls, hidden_dim,
                                 embedding_dim, encoding_dim, batch_size, lr)

if __name__ == "__main__":

    print('CPU count:', multiprocessing.cpu_count())

    # Start the grid search using multiprocessing
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(multiprocessing_wrapper, all_combinations)

    # Close the pool
    pool.close()
    pool.join()

    # Find the best model
    best_model_info = {
        'train_loss': float('inf'),
        'attack_loss': float('-inf'),
        'hidden_dim': None,
        'embedding_dim': None,
        'encoding_dim': None,
        'batch_size': None,
        'learning_rate': None
    }

    for result in results:
        train_loss, attack_loss, hidden_dim, embedding_dim, encoding_dim, batch_size, lr = result
        if train_loss < best_model_info['train_loss'] and attack_loss > best_model_info['attack_loss']:
            best_model_info.update({
                'train_loss': train_loss,
                'attack_loss': attack_loss,
                'hidden_dim': hidden_dim,
                'embedding_dim': embedding_dim,
                'encoding_dim': encoding_dim,
                'batch_size': batch_size,
                'learning_rate': lr
            })

    print("Best Model:", best_model_info)
