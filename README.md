# FlashGEMM-v3
Implement a **general FP16/BF16 GEMM + Softmax fusion operator** on **NVIDIA GPU**, integrate it with **ONNX-Runtime backend** via **TVM Auto-Tuning**, and finally provide **end-to-end accuracy verification report**



## 2 详细计划（含每日 checklist）

### Week 1 需求拆解 & 算子分析

- **Day1** 选模型：ViT-B/16（CV）+ Whisper Encoder（语音）+ BERT-Large（NLP）+ DLRM（推荐）  
    用 PyTorch 导出 ONNX → Netron 打开 → 统计 MatMul+Softmax 维度分布。
    
- **Day2** 定义算子接口：`fused_gemm_softmax(M,N,K, A,B, bias, mask)`，支持 FP16/BF16、mask 可选。
    
- **Day3** 精度基线：ONNXRuntime CUDA EP 跑 FP32 作为 golden。
    
- **Day4** Roofline 分析：Nsight Compute → 计算算术强度、带宽上限。
    
- **Day5** 需求冻结：输出《FlashGEMM-V3 SPEC》markdown。
    

### Week 2 CUDA 核心算子开发

- **Day6-7** 手写 FP16 GEMM tile-based kernel（128×128×32）
    
    - 使用 shared memory double buffer
        
    - warp-level MMA (`mma.sync`)
        
- **Day8** Softmax 融合：GEMM 输出 tile 直接在 shared memory 做 row-wise softmax，避免写回 global。
    
- **Day9** 支持 causal mask（NLP 用）→ 条件 load mask 常量到 register。
    
- **Day10** 单元测试：pytest + cupy.allclose 误差 < 1e-3。
    

### Week 3 Triton 版本 & 性能对比

- **Day11-12** 用 Triton DSL 重写同一算子 < 100 行。
    
- **Day13** Triton Autotuner 搜索 tile、num_stages、num_warps。
    
- **Day14** 性能打擂台：CUDA vs Triton vs cuBLAS+Softmax 分离；记录 GFLOPs 与带宽。
    

### Week 4 Auto-Tuning & 低比特实验

- **Day15-16** TVM Meta-Schedule：
    
    - 定义 TensorIR schedule：blockIdx/blockIdx k-split + shared memory cache。
        
    - 跑 1024 random schedules → 选 top-10。
        
- **Day17** BF16 路径：修改 PTX `.target sm_89` + `__nv_bfloat16` 类型。
    
- **Day18** INT8 量化尝试：cutlass 3.0 `GemmUniversal` + `FastInterleavedAndBiased` epilogue，记录精度下降。
    

### Week 5 编译集成 & 框架对接

- **Day19** ONNXRuntime Custom EP：
    
    - 新建 `flash_gemm_softmax_op.cc` 注册 kernel。
        
    - CMake 链接你的 `.so`。
        
- **Day20** TensorFlow plugin：TF-Custom-Op-CC 模板，注册 `FusedGemmSoftmax`。
    
- **Day21** 端到端测试：
    
    - ViT-B 推理 batch=16，验证 top-1 误差 ≤ 0.05 %。
        
    - Nsight Systems 查看 kernel 融合后 global memory traffic ↓ 30%。
        

### Week 6 精度验证 & 项目收官

- **Day22** 误差分析脚本：
    
    Python
    
    复制
    
    ```python
    np.testing.assert_allclose(golden, actual, rtol=1e-3, atol=1e-4)
    ```
    
- **Day23** Corner case 测试：K=1、M=1、N=65536、带 dropout mask。
    
- **Day24** 性能回归：用 AirSpeed Velocity 写 benchmark，CI 触发。
    
- **Day25** 撰写 README：架构图 + 调优曲线 + 精度表。
    
- **Day26** 内部 tech talk（15 min），录屏上传。
    
- **Day27-28** Buffer days，处理 code review。
    
- **Day29** Tag v1.0，打 release。
    
- **Day30** Retro：列 3 个 next step（Flash-Attention3、Hopper TMA、CPython wheel）。





## 3 文件详细

- day1 
    getmodel.py  
    生成 bert_large.onnx 
    生成 whisper_encoder.onnx  
    模型文件


