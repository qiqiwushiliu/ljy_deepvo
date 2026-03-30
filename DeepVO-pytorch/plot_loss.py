import matplotlib.pyplot as plt
import numpy as np
import os
import json
from config.params import par


def load_loss_from_json(json_path):
    """Load training loss data from JSON file.

    Returns:
        epochs: list of epoch numbers (1-indexed)
        train_list: list of train losses
        valid_list: list of valid losses
    """
    if not os.path.exists(json_path):
        print(f'Loss JSON file not found: {json_path}')
        return None, None, None

    with open(json_path, 'r') as f:
        loss_data = json.load(f)

    if not loss_data.get('train_loss'):
        print(f'No loss data found in {json_path}')
        return None, None, None

    epochs = list(range(1, len(loss_data['train_loss']) + 1))
    return epochs, loss_data['train_loss'], loss_data['valid_loss']


def plot_loss(epochs, train_loss, valid_loss, title, save_path):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, valid_loss, 'r-', label='Valid Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Saved: {save_path}')
    plt.close()


def main():
    # Create result directory if not exists
    results_loss_dir = '/home/LXT/LJY/DeepVO-pytorch/results/loss'
    if not os.path.exists(results_loss_dir):
        os.makedirs(results_loss_dir)

    models = [
        ('_cfc', 'DeepVO (CfC/NCP)'),
        ('_lstm', 'DeepVO (LSTM)'),
    ]

    all_data = {}

    for suffix, name in models:
        par.model_suffix = suffix
        loss_json_path = par.get_loss_json_path()

        print(f'\nProcessing {name}...')
        print(f'Loss JSON path: {loss_json_path}')

        epochs, train_loss, valid_loss = load_loss_from_json(loss_json_path)

        if epochs is None:
            print(f'Skipping {name} - no data')
            continue

        all_data[suffix] = (epochs, train_loss, valid_loss, name)

        # Plot individual
        title = f'{name} - Training and Validation Loss'
        save_path = f'{results_loss_dir}/loss{suffix}.png'
        plot_loss(epochs, train_loss, valid_loss, title, save_path)

        # Print final
        print(f'{name} - Final train: {train_loss[-1]:.6f}, valid: {valid_loss[-1]:.6f}')

    # Plot comparison
    if len(all_data) >= 2:
        plt.figure(figsize=(12, 6))
        colors = {'_cfc': 'blue', '_lstm': 'red'}

        for suffix, (epochs, train_loss, valid_loss, name) in all_data.items():
            plt.plot(epochs, train_loss, color=colors.get(suffix, 'blue'), linestyle='-',
                     label=f'{name} - Train', linewidth=2)
            plt.plot(epochs, valid_loss, color=colors.get(suffix, 'blue'), linestyle='--',
                     label=f'{name} - Valid', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_loss_dir}/loss_comparison.png')
        print(f'\nSaved: {results_loss_dir}/loss_comparison.png')
        plt.close()
    elif len(all_data) == 1:
        print('\nOnly one model has data, skipping comparison.')
    else:
        print('\nNo data found for any model.')

    print('\nDone!')


if __name__ == '__main__':
    main()