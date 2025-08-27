









import torch
from typing import Optional
import torch.nn.functional as F

def fused_gemm_softmax(
    M:int,
    N:int, 
    K:int,
    A:torch.tensor, 
    B:torch.tensor,
    bias:Optional[torch.tensor]=None, 
    mask:Optional[torch.tensor]=None,
    dtype:Optional[torch.dtype]=None
    ) -> torch.tensor:

    """
    Fused GEMM + Softmax operation.

    Parameters:
    - M (int): Number of rows in matrix A and output.
    - N (int): Number of columns in matrix B and output.
    - K (int): Number of columns in matrix A and rows in matrix B.
    - A (torch.Tensor): Input tensor of shape (M, K).
    - B (torch.Tensor): Input tensor of shape (K, N).
    - bias (torch.Tensor, optional): Bias tensor of shape (N,). Default is None.
    - mask (torch.Tensor, optional): Mask tensor of shape (M, N). Default is None.

    Returns:
    - torch.Tensor: Output tensor after applying fused GEMM and Softmax.
    """

    # Input validation
    assert A.shape == (M, K), f"Expected A shape {(M, K)}, but got {A.shape}"
    assert B.shape == (K, N), f"Expected B shape {(K, N)}, but got {B.shape}"

    # bias shape check
    if bias is not None:
        if bias.dim() == 1:
            assert bias.shape == (N,), f"Expected bias shape {(N,)}, but got {bias.shape}"
        elif bias.dim() == 2:
            assert bias.shape == (M.N), f"Expected bias shape {(M,N)}, but got {bias.shape}"
        else:
            raise ValueError(f"Bias must be 1D or 2D tensor, but got {bias.dim()}D tensor")
        
    # mask shape check
    if mask is not None:
        assert mask.shape == (M, N), f"Expected mask shape {(M, N)}, but got {mask.shape}"

    # set target dtype
    target_dtype = dtype if dtype is not None else A.dtype

    # Cast A and B to target dtype
    A = A.to(target_dtype)
    B = B.to(target_dtype)
    if bias is not None:
        bias = bias.to(target_dtype)

    # Step1 GEMM operation
    # C = A@B
    C = torch.matmul(A, B)  # Shape: (M, N)

    # Step2 Add bias if provided
    if bias is not None:
        if bias.dim() == 1:
            C += bias.unsqueeze(0)  
        else:
            C += bias

    # step3 Apply mask if provided
    if mask is not None:
        mask_value = -float('inf') if target_dtype in [torch.float32, torch.float16] else -float(65504)
        C = C.masked_fill(mask == 0, mask_value)
    
    # Step4 Softmax operation
    output = F.softmax(C, dim=-1)

    return output



def create_casual_mask(
        M:int, 
        N:int, 
        device:Optional[torch.device]=None,
        ) -> torch.tensor:
    """
    Create a causal mask for attention mechanisms.

    Parameters:
    - M (int): Number of rows in the mask.
    - N (int): Number of columns in the mask.
    - device (torch.device, optional): Device to create the mask on. Default is None.

    Returns:
    - torch.Tensor: Causal mask tensor of shape (M, N).
    """
    # no device specified, use cpu
    if device is None:
        device = torch.device('cuda:0')

    # Create a lower triangular matrix
    mask = torch.tril(torch.ones((M, N), device=device))

    return mask.float()






