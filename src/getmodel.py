

## Vit



from transformers import WhisperModel

import torch

'''
 // 导入whisper-base，export onnx
'''
# model = WhisperModel.from_pretrained("openai/whisper-base",cache_dir='../models').encoder.eval()
# dummy = torch.randn(1,80,3000)

# torch.onnx.export(
#     model,dummy,"whisper_encoder.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=14
# )




'''
 // 导入Bert ，export onnx
'''
# from transformers import BertModel


# model = BertModel.from_pretrained("bert-large-uncased",cache_dir='../models').eval()

# dummy = torch.randint(0,30522,(1,128))

# torch.onnx.export(
#     model,dummy,"bert_large.onnx",
#     input_names =["input_ids"],
#     output_names=["output"],
#     opset_version=14
# )



'''
// 打印输出
'''


from onnx import load

model_path =  "../models/bert_large.onnx"
onnx_model = load(model_path)
graph = onnx_model.graph

for node in graph.node:
    if node.op_type in ["MatMul", "Softmax"]:
        in_shape = [graph.get_tensor_shape(t) for t in node.input]
        out_shape = [graph.get_tensor_shape(node.output[0])]
        print(f"node_name: {node.name}   node_type: {node.op_type}    in: {in_shape}     out: {out_shape}")
        print("----")
