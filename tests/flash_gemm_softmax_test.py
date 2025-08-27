



import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from flash_gemm_softmax import fused_gemm_softmax,create_casual_mask


# Test utilities

def test_basic_functionality():

    """Basic functionality test for fused_gemm_softmax."""
    M,N,K = 4,6,8
    A = torch.randn(M,K, device='cuda', dtype=torch.float16)
    B = torch.randn(K,N, device='cuda', dtype=torch.float16)

    ## Without bias and mask
    result1 = fused_gemm_softmax(M,N,K,A,B)
    print(f"Test 1 - Shape: {result1.shape}, Sum per row: {result1.sum(dim=1)}")
    print(f"Result 1: {result1}")
    print("----------------------------------------------------------------------")

    ## With bias 
    bias = torch.randn(N, device='cuda', dtype=torch.float16)
    result2 = fused_gemm_softmax(M,N,K,A,B,bias=bias)
    print(f"Test 2 - Shape: {result2.shape}, Sum per row:{result2.sum(dim=1)}")
    print(f"Result 2: {result2}")
    print("----------------------------------------------------------------------")

    ## With causal mask
    mask = create_casual_mask(M,N)
    result3 = fused_gemm_softmax(M,N,K,A,B,mask=mask)
    print(f"Test 3 - Shape: {result3.shape}, Sum per row:{result3.sum(dim=1)}")
    print(f"Mask:\n {mask}")
    print(f"Result 3: {result3}")
    print("----------------------------------------------------------------------")


def test_precision_modes():

    """Test diffrerent precision modes for fused_gemm_softmax."""
    M,N,K = 8,8,16
    A = torch.randn(M,K, device='cuda', dtype=torch.float16)
    B = torch.randn(K,N, device='cuda', dtype=torch.float16)

    ## fp32 test
    result_fp32 = fused_gemm_softmax(M,N,K,A,B,dtype=torch.float32)

    ## fp16 test
    result_fp16 = fused_gemm_softmax(M,N,K,A,B,dtype=torch.float16)

    ## bf16 test
    if torch.cuda.is_available():
        result_bf16 = fused_gemm_softmax(M,N,K,A,B,dtype=torch.bfloat16)
        print(f"BF16 Test Passed! - Shape: {result_bf16.shape}, Sum per row:{result_bf16.sum(dim=1)}")

    print(f"FP32 Test Passed! - Shape: {result_fp32.shape}, Sum per row:{result_fp32.sum(dim=1)}")
    print(f"FP16 Test Passed! - Shape: {result_fp16.shape}, Sum per row:{result_fp16.sum(dim=1)}")

    #check numerical difference
    diff = torch.abs(result_fp32 - result_fp16.float()).max().item()
    print(f"Max absolute difference between FP32 and FP16: {diff}")



if __name__ == "__main__":

    # run tests
    test_basic_functionality()
    test_precision_modes()





