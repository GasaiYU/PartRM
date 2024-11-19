import torch
import torch.nn as nn

from torchvision.transforms import GaussianBlur

class FourierEmbedder(object):
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)


class DragPositionNet(nn.Module):
    def __init__(self, num_drags, fourier_freqs=8, downsample_ratio=64):
        super().__init__()
        self.num_drags = num_drags

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*2 # 2 for sin and cos, 2 for 2 dims (x1, y1） or (x2, y2)
        
        # -------------------------------------------------------------- #
        self.linears_drag = nn.Sequential(
            nn.Linear(self.position_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )
        
        self.downsample_ratio = downsample_ratio
    

    def forward(self, drags_start, drags_end):
        # drags_start: [B, V, N, 2], start points of drags
        # drags_end: [B, V, N, 2], move vectors of drags
        B, V, N, _ = drags_start.shape
        drags_start = drags_start.view(B*V, N, -1)
        
        drags_start_embeddings = []
        for i in range(N):
            drag_start_embedding = self.fourier_embedder(drags_start[:, i, :])
            drags_start_embeddings.append(self.linears_drag(drag_start_embedding))
        drags_start_embeddings = torch.stack(drags_start_embeddings, dim=1)
        
        drags_end = drags_end.view(B*V, N, -1)
        drags_end_embeddings = []
        for i in range(N):
            drag_end_embedding = self.fourier_embedder(drags_end[:, i, :])
            drags_end_embeddings.append(self.linears_drag(drag_end_embedding))
        drags_end_embeddings = torch.stack(drags_end_embeddings, dim=1)
        
        merge_start_embeddings = torch.zeros((B*V, 512, 8, 8)).to(drag_start_embedding.device) # [B*V, 256, 8, 8]
        merge_end_embeddings = torch.zeros((B*V, 512, 8, 8)).to(drag_start_embedding.device) # [B*V, 256, 8, 8]

        for i in range(B*V):
            for j in range(N):
                merge_start_embeddings[i, :, int(drags_start[i, j, 0]) // self.downsample_ratio, 
                                 int(drags_start[i, j, 1]) // self.downsample_ratio] += drags_start_embeddings[i,j,:]
                merge_end_embeddings[i, :, int(drags_end[i, j, 0]) // self.downsample_ratio, 
                                 int(drags_end[i, j, 1]) // self.downsample_ratio] += drags_end_embeddings[i,j, :]

        merge_embeddings = torch.cat([merge_start_embeddings, merge_end_embeddings], dim=1)
        return merge_embeddings
    

class DragPositionNetMultiScale(nn.Module):
    def __init__(self, fourier_freqs=8, scales=[256, 128, 64, 32, 16, 8], channels=[64, 64, 128, 256, 512, 1024], drag_layer_idx=None):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*2 # 2 for sin and cos, 2 for 2 dims (x1, y1） or (x2, y2)
        
        # -------------------------------------------------------------- #
        self.linear_drags = []
        for i in range(len(channels)):
            if drag_layer_idx is not None and i != drag_layer_idx:
                continue
            self.linear_drags.append(nn.Sequential(
                nn.Linear(self.position_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 256),
                nn.SiLU(),
                nn.Linear(256, channels[i]//2),
            ))
            
        self.linears_drags = nn.ModuleList(self.linear_drags)
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)
        
        self.scales = scales
        self.channels = channels
        
        if drag_layer_idx is not None:
            self.drag_layer_idx = drag_layer_idx
            self.scales = self.scales[self.drag_layer_idx:self.drag_layer_idx+1]
            self.channels = self.channels[self.drag_layer_idx:self.drag_layer_idx+1]
        
    def forward(self, drags_start, drags_end):
        scales = self.scales
        channels = self.channels
        
        B, V, N, _ = drags_start.shape  
        
        drags_start = drags_start.view(B*V, N, -1)
        drags_end = drags_end.view(B*V, N, -1)


        multi_scale_merge_start_embeddings = []
        multi_scale_merge_end_embeddings = []
        
        for idx, scale in enumerate(scales):
            drags_start_embeddings = []
            drags_end_embeddings = []
            for i in range(N):
                drag_start_embedding = self.fourier_embedder(drags_start[:, i, :])
                drags_start_embeddings.append(self.linears_drags[idx](drag_start_embedding))
            drags_start_embeddings = torch.stack(drags_start_embeddings, dim=1)
            
            for i in range(N):
                drag_end_embedding = self.fourier_embedder(drags_end[:, i, :])
                drags_end_embeddings.append(self.linears_drags[idx](drag_end_embedding))
            drags_end_embeddings = torch.stack(drags_end_embeddings, dim=1)
        
            merge_start_embeddings = torch.zeros((B*V, channels[idx]//2, scale, scale)).to(drag_start_embedding.device)
            merge_end_embeddings = torch.zeros((B*V, channels[idx]//2, scale, scale)).to(drag_start_embedding.device)
            downsample_ratio = 512 // scale
            
            for i in range(B*V):
                for j in range(N):
                    merge_start_embeddings[i, :, int(drags_start[i, j, 0]) // downsample_ratio, 
                                     int(drags_start[i, j, 1]) // downsample_ratio] += drags_start_embeddings[i,j,:]
                    merge_end_embeddings[i, :, int(drags_end[i, j, 0]) // downsample_ratio, 
                                     int(drags_end[i, j, 1]) // downsample_ratio] += drags_end_embeddings[i,j, :]
                    # merge_end_embeddings[i, :, int(drags_start[i, j, 0]) // downsample_ratio, 
                    #                 int(drags_start[i, j, 1]) // downsample_ratio] += drags_end_embeddings[i,j, :]
            # Add Gaussian Blur
            merge_start_embeddings = self.gaussian_blur(merge_start_embeddings)
            merge_end_embeddings = self.gaussian_blur(merge_end_embeddings)
            
            multi_scale_merge_start_embeddings.append(merge_start_embeddings)
            multi_scale_merge_end_embeddings.append(merge_end_embeddings)
        
        return multi_scale_merge_start_embeddings, multi_scale_merge_end_embeddings
            