
import torch
from typing import Optional,List,Dict
import numpy as np
import onnxruntime as ort
from flash_gemm_softmax import fused_gemm_softmax,benchmark_shapes_from_models
import os
from datetime import datetime

"""
Gemm + Softmax Module 
"""
class GemmSoftmaxModule(torch.nn.Module):
    def __init__(self,has_bias:bool, has_mask:bool):
        super(GemmSoftmaxModule, self).__init__()
        self.has_bias = has_bias
        self.has_mask = has_mask

    def forward(
            self, 
            A:torch.Tensor, 
            B:torch.Tensor,
            bias:Optional[torch.Tensor]=None, 
            mask:Optional[torch.Tensor]=None
            ) -> torch.Tensor:
        
        # Step1 GEMM operation
        C = torch.matmul(A, B)

        # Step2 Add bias if provided
        if self.has_bias and bias is not None:
            if bias.dim() == 1:
                C += bias.unsqueeze(0)  
            else:
                C += bias
        
        # Step3 Apply mask if provided
        if self.has_mask and mask is not None:
            mask_value = torch.finfo(C.dtype).min
            C = C.masked_fill(mask == 0, mask_value)
        
        # Step4 Softmax operation
        output = torch.nn.functional.softmax(C, dim=-1)

        return output
        

class GoldenBaseline:
    
    def __init__(self,use_cuda:bool = True):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        self.result = {}

        if use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA is not available, falling back to CPU.")
            self.providers = ['CPUExecutionProvider']
            raise ValueError("CUDA is not available, please set use_cuda to False.")
        
    def create_gemm_softmax_onnx(
        self,
        M:int,
        N:int, 
        K:int,
        has_bias:bool=True,
        has_mask:bool=False,
        ) -> str:
        """
        Parameters:
        - M (int): Number of rows in matrix A and output.
        - N (int): Number of columns in matrix B and output.
        - K (int): Number of columns in matrix A and rows in matrix B.
        - has_bias (bool): Whether to include bias addition.
        - has_mask (bool): Whether to include masking before softmax.
        Returns:
        - torch.Tensor: Output tensor after applying fused GEMM and Softmax.
        """
    
        # Create model instance and set to eval mode
        model = GemmSoftmaxModule(has_bias, has_mask)
        model.eval()

        # Prepare dummy inputs with fp32
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        input = [A, B]
        input_names = ["A", "B"]

        # Add bias and mask if needed
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            input.append(bias)
            input_names.append("bias")
        if has_mask:
            mask = torch.tril(torch.ones((M, N), dtype=torch.float32))
            input.append(mask)
            input_names.append("mask")

        onnx_path = f"gemm_softmax_M{M}_N{N}_K{K}_bias{has_bias}_mask{has_mask}.onnx"

        # Export the model to ONNX
        torch.onnx.export(
            model,
            tuple(input),
            onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={
                "A":{0:"M"},
                "B":{1:"N"},
                "output":{0:"M", 1:"N"}
            } if not has_mask else None,
            opset_version=17,
            do_constant_folding=True
        )

        return onnx_path,input
    
    def run_onnx_inference(
        self,
        onnx_path:str,
        input:List[torch.tensor] = None,
        ) -> np.ndarray:

        """
        - onnx_path (str): Path to the ONNX model file.
        - input (List[torch.tensor]): List of input tensors.
        """

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=self.providers)

        # Prepare input dictionary
        input_dict = {}
        for i, input_tensor in enumerate(input):
            input_name = session.get_inputs()[i].name
            input_dict[input_name] = input_tensor.cpu().numpy().astype(np.float32)

        # Run inference
        outputs = session.run(None, input_dict)

        return outputs[0]
    
    def run_pytorch_inference(
        self,
        M:int,
        N:int,
        K:int,
        A:torch.Tensor,
        B:torch.Tensor,
        bias:Optional[torch.Tensor]=None,
        mask:Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        
        return fused_gemm_softmax(M,N,K,A,B,bias,mask)
    
    @staticmethod
    def compare_results(
        onnx_result:np.ndarray,
        pytorch_result:torch.Tensor,
        tolerance:float=1e-5
        ) -> dict:

        """
        Compare the results bwtween onnx and pytorch
        """

        pytorch_np = pytorch_result.cpu().numpy()

        # Calculate absolute and relative differences
        abs_diff = np.abs(onnx_result - pytorch_np)
        rel_diff = abs_diff / (np.abs(pytorch_np) + 1e-8)

        # Check all resultes
        comparison = {
            "max_abs_error": np.max(abs_diff),
            "max_rel_error": np.max(rel_diff),
            "mean_abs_error": np.mean(abs_diff),
            "mean_rel_error": np.mean(rel_diff),
            "tolerance_used": tolerance,
            'close_within_tolerance': np.allclose(onnx_result, pytorch_np, 
                                    atol=tolerance, rtol=tolerance),
        }

        return comparison

    def benchmark_single_case(
        self,
        M:int,
        N:int,
        K:int,
        has_bias:bool=True,
        has_mask:bool=False,
        ) -> Dict:

        """
        Benchmark test for a single case of Gemm + Softmax
        """
        print(f"Testing case: M={M}, N={N}, K={K}, bias={has_bias}, mask={has_mask}")
        onnx_path,input = self.create_gemm_softmax_onnx(M,N,K,has_bias,has_mask)
        onnx_result = self.run_onnx_inference(onnx_path,input)

        # Get A B
        A, B = input[0], input[1]
        bias = input[2] if has_bias else None
        mask = input[3] if has_mask else None

        # Calculate pytorch result
        pytorch_result = self.run_pytorch_inference(M,N,K,A,B,bias,mask)

        # Compare results
        comparison = self.compare_results(onnx_result, pytorch_result)

        # Clean up the onnx file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        # Store results
        results = {
            "shape": (M,N,K),
            "has_bias": has_bias,
            "has_mask": has_mask,
            "onnx_result_shape": onnx_result.shape,
            "pytorch_result_shape": pytorch_result.shape,
            "comparison": comparison,
            "providers": self.providers
        }

        return results
    
    def run_full_benchmark(self)->Dict:
        """
        Run full benchmark suite with various configurations
        """
        print("Building Golden Baseline with ONNXRuntime CUDA EP")

        all_results = []
        shapes = benchmark_shapes_from_models()

        test_configs = [
            (True, False),
            (True, True),  # causal
            (False, False),
        ]

        # Loop over all shapes and configurations
        for M,N,K in shapes:
            for has_bias, has_mask in test_configs:
                try:
                    result = self.benchmark_single_case(M,N,K,has_bias,has_mask)
                    all_results.append(result)

                    comp = result["comparison"]
                    status = "✅ PASS" if comp['close_within_tolerance'] else "❌ FAIL"
                    print(f"  {status} | Max abs err: {comp['max_abs_error']:.2e} | "
                          f"Max rel err: {comp['max_rel_error']:.2e}")

                except Exception as e:
                    print(f"Error testing case M={M}, N={N}, K={K}, bias={has_bias}, mask={has_mask}: {e}")
                    continue

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(all_results),
            "passed_tests": sum(1 for r in all_results if r["comparison"]["close_within_tolerance"]),
            "providers": self.providers,
            "detailed_results": all_results
        }

        return summary

    def save_golden_baseline(self,filepath:str="../results/golden_baseline_results.json",results:Dict=None):
        """
        Save the golden baseline results to a JSON file
        """
        import json     
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else int(x) if isinstance(x, (np.integer,)) else str(x))
        print(f"Golden baseline results saved to {filepath}")   

    def print_summary(self,results:Dict=None):
        """
        Print a summary of the benchmark results
        """
        print("\n" + "=" * 60)
        print("GOLDEN BASELINE SUMMARY")
        print("=" * 60)
        
        total = results['total_tests']
        passed = results['passed_tests']
        print(f"Total tests: {total}")
        print(f"Passed tests: {passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print(f"Execution providers: {results['providers']}")

        if results['detailed_results']:
            max_errs = [r['comparison']['max_abs_error'] for r in results['detailed_results']]
            print(f"Max absolute error range: {min(max_errs):.2e} - {max(max_errs):.2e}")

        print(f"Timestamp: {results['timestamp']}")
        print("=" * 60)

def main():
    # check onnxruntime installed   
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available Execution Providers: {ort.get_available_providers()}")
    except ImportError:
        raise ImportError("ONNX Runtime is not installed. Run 'pip install onnxruntime-gpu' to install it.")
    
    baseline = GoldenBaseline(use_cuda=True)
    results = baseline.run_full_benchmark()
    baseline.print_summary(results)
    baseline.save_golden_baseline(results=results)
    


if __name__ == "__main__":
    main()

    


    






