import torch

log_var = torch.tensor(
			[1,1,1,1,1],
			dtype=None,
			device=None,
			requires_grad=False,
			pin_memory=False
			)

sigma = torch.exp(log_var * 0.5)
eps = torch.randn_like(sigma)
print(sigma)
print(eps)  
