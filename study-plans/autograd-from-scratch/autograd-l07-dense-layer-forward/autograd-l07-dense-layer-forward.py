import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    ans =  inputs @ weight_matrix.T + biases 

    if nonlinear:
        ans = torch.tanh(ans)

    return ans
