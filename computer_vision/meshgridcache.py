import torch

class MeshgridCache:
    _instance = None
    _grids = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MeshgridCache, cls).__new__(cls)
        return cls._instance

    def get_meshgrid(self, shape, device='cuda:0'):
        if shape not in self._grids:
            H, W = shape
            u = torch.arange(W, device=device)
            v = torch.arange(H, device=device)
            # Adjust indexing based on PyTorch's meshgrid behavior
            v_grid, u_grid = torch.meshgrid(v, u)
            self._grids[shape] = torch.stack((u_grid, v_grid), dim=-1)
        return self._grids[shape]

