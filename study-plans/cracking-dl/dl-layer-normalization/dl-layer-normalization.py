import numpy as np

def layer_normalization(x: list, gamma: list, beta: list, eps: float = 1e-05, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    # Force high-precision float64 arrays immediately upon conversion
    x = np.array(x, dtype=np.float64)
    gamma = np.array(gamma, dtype=np.float64)
    beta = np.array(beta, dtype=np.float64)
    n = x.shape[-1]

    x_mean = np.mean(x, axis = -1, keepdims=True)
    x_var = np.var(x, axis = -1, keepdims=True)
    x_hat = ( x - x_mean) / np.sqrt(x_var + eps)

    y = gamma * x_hat + beta

    ans = {'output': np.round(y, decimals = 4).tolist(),'mean': [round(float(v), 4) for v in x_mean.ravel()], 'var': np.round(np.squeeze(x_var, axis = -1), decimals=4).tolist(), 'x_hat': np.round(x_hat, decimals = 4).tolist()}

    if mode == "forward":
        return ans
    elif mode == "backward":
        d_output = np.array(d_output, dtype=np.float64)
        
        # 2. Parameter Gradients (Summing along the batch dimension axis=0)
        grad_beta = np.sum(d_output, axis=0)
        grad_gamma = np.sum(x_hat * d_output, axis=0)
        
        # 3. Input Gradient Calculation
        grad_x_hat = gamma * d_output
        inv_std = 1 / np.sqrt(x_var + eps)
        
        # Fixed the final variance sum multiplication term (grad_x_hat * x_hat)
        grad_x = (inv_std / n) * (
            n * grad_x_hat - 
            np.sum(grad_x_hat, axis=-1, keepdims=True) - 
            x_hat * np.sum(grad_x_hat * x_hat, axis=-1, keepdims=True)
        )

        # Convert backprop arrays to list outputs rounded to 4 decimals
        ans['dgamma'] = np.round(grad_gamma, decimals=4).tolist() # Note: mapping keys as required by your platform schema
        ans['dbeta'] = np.round(grad_beta, decimals=4).tolist()
        ans['dx'] = np.round(grad_x, decimals=4).tolist()
    
    return ans
