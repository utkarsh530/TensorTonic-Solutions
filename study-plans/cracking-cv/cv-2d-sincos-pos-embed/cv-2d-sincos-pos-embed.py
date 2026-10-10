import numpy as np

def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> list:
    """
    Returns a float list of shape (grid_h * grid_w, embed_dim), rounded to 4 decimals.
    """
    output = np.zeros((grid_h, grid_w, embed_dim))

    d = embed_dim // 2
    for i in range(grid_h):
        for j in range(grid_w):
            ft_hf = np.array([np.sin(i * (10000 ** (-2*k/d))) for k in range(0, d // 2)])
            sd_hf = np.array([np.cos(i * (10000 ** (-2*k/d))) for k in range(0, d // 2)])
            ft =  np.concatenate((ft_hf, sd_hf))

            ft_hf = np.array([np.sin(j * (10000 ** (-2*k/d))) for k in range(0, d // 2)])
            sd_hf = np.array([np.cos(j * (10000 ** (-2*k/d))) for k in range(0, d // 2)])
            sd = np.concatenate((ft_hf, sd_hf))

            output[i,j,:] = np.round(np.concatenate((ft, sd)), decimals = 4)


    # print(output)
        
    return np.reshape(output, (grid_h * grid_w, embed_dim)).tolist()