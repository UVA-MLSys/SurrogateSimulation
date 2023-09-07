# Step 1: Import the required packages
import torch

def array_to_tensor(arr):
    """Convert a 2D array to a PyTorch tensor."""
    return torch.tensor(arr)

# Sample 2D array
two_dim_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

tensor_result = array_to_tensor(two_dim_array)
print(tensor_result)