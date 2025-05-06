from pathlib import Path

def get_config():  # Defines all key settings and training parameters
    return {
        "batch_size": 8,  # Number of samples per training batch
        "num_epochs": 20,  # Total number of training epochs
        "lr": 10**-4,  # Learning rate
        "seq_len": 350,  # Maximum sequence length (in tokens)
        "d_model": 512,  # Embedding/hidden size in Transformer layers
        "nums_head":8,
        "datasource": "wmt14",  # Dataset name used for language translation
        "lang_src": "cs",  # Source language (Czech)
        "lang_tgt": "en",  # Target language (English)
        "model_folder": "weights",  # Folder to save model weights
        "model_basename": "tmodel_",  # Base name for model files
        "preload": "latest",  # Whether to preload "latest" or a specific checkpoint
        "tokenizer_file": "tokenizer_{0}.json",  # Format for tokenizer filename
        "experiment_name": "runs/tmodel"  # TensorBoard experiment log directory
    }

def get_weights_file_path(config, epoch: str):
    # Constructs the full file path for the weights of a given epoch
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
    # Example: './wmt14_weights/tmodel_5.pt'

def latest_weights_file_path(config):
    # Finds the most recent model checkpoint (latest .pt file)
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    
    if len(weights_files) == 0:
        return None  # No saved weights found
    
    weights_files.sort()  # Sort by name (relies on epoch numbers in filenames)
    return str(weights_files[-1])  # Return the path to the latest checkpoint
