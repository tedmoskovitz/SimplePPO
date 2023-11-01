import os
import flax
from flax import serialization
from typing import Any, Optional

class CheckpointManager:
    def __init__(self, directory: str, run_name: Optional[str] = None):
        self.directory = directory
        self.run_name = run_name if run_name else "checkpoint"
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def save(self, train_state: Any, iteration: int, config: Optional[dict] = None):
        path = os.path.join(self.directory, f"{self.run_name}_{iteration}.flax")
        with open(path, 'wb') as file:
            serialized_data = serialization.to_bytes(train_state)
            file.write(serialized_data)
        print(f"Checkpoint saved at {path}")
    
    def load_latest(self):
        checkpoints = [file for file in os.listdir(self.directory) if file.startswith("checkpoint_")]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
        path = os.path.join(self.directory, latest_checkpoint)
        with open(path, 'rb') as file:
            serialized_data = file.read()
            train_state = serialization.from_bytes(train_state, serialized_data) # Assuming you pass in a blank train_state
        print(f"Loaded checkpoint from {path}")
        return train_state


def load_checkpoint(file_path, model, optimizer_state=None):
    """
    Load a checkpoint for the agent.

    Args:
        file_path (str): Path to the checkpoint file to load.
        model (flax.linen.Module): The Flax model for which to restore the parameters.
        optimizer_state (optax.State, optional): If provided, optimizer state to restore.

    Returns:
        restored_params (Any): Restored model parameters.
        restored_state (optax.State, optional): Restored optimizer state.
    """
    with open(file_path, 'rb') as f:
        restored = flax.serialization.from_bytes(model, f.read())
    
    if optimizer_state:
        restored_params, restored_state = restored
        return restored_params, restored_state
    return restored


