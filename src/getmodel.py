

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
from transformers import BertModel


model = BertModel.from_pretrained("bert-large-uncased",cache_dir='../models').eval()

dummy = torch.randint(0,30522,(1,128))

torch.onnx.export(
    model,dummy,"bert_large.onnx",
    input_names =["input_ids"],
    output_names=["output"],
    opset_version=14
)


