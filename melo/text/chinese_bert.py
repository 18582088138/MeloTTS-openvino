import torch
import sys
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.onnx import FeaturesManager

import openvino as ov
import os
import nncf
from functools import partial
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


# model_id = 'hfl/chinese-roberta-wwm-ext-large'
local_path = "./bert/chinese-roberta-wwm-ext-large"


tokenizers = {}
models = {}
core = ov.Core()

class CustomDataset(Dataset):
    def __init__(self, data_count=100, dummy_data=None):
        self.dataset = []
        print("====",dummy_data)
        for idx, tmp_data in enumerate(dummy_data):
            print("====",tmp_data)
        for i in range(data_count):
            self.dataset.append([dummy_data])
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        image = self.dataset[idx]
        return image

def transform_fn(data_item):
    data_item = data_item[0] 
    inputs = {"input_ids":data_item["input_ids"].squeeze(0),
              "token_type_ids":data_item["token_type_ids"].squeeze(0),
              "attention_mask":data_item["attention_mask"].squeeze(0),               
            }
    return inputs

def onnx_model_convert(onnx_model_path, torch_model, tokenizer,feature="sequence-classification"):
    print("======Onnx Model Convert=======")
    if not os.path.exists(onnx_model_path):
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(torch_model, feature=feature)
        onnx_config = model_onnx_config(torch_model.config)
        onnx_inputs, onnx_outputs = transformers.onnx.export(
            preprocessor=tokenizer,
            model=torch_model,
            config=onnx_config,
            opset=13,
            output=Path(onnx_model_path)
            )

def ov_model_convert(onnx_model_path, sample_input, ov_model_path="ZH_bert.xml"):        
    print("=====OpenVINO Model Convert=====")
    if not os.path.exists(ov_model_path):
        ov_model = ov.convert_model(onnx_model_path,example_input=sample_input)
        ov.save_model(ov_model, ov_model_path)
        print(f"== OV {ov_model_path} convert success ==")
    else:
        ov_model = core.read_model(ov_model_path)
    return ov_model

def ov_nncf_quant(ov_model, sample_input, ov_int8_model_path):
    print("=====OpenVINO NNCF Model Quantization=====")
    if not os.path.exists(ov_int8_model_path):
        custom_dataset = CustomDataset(data_count=100, dummy_data=sample_input)
        calibration_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False)
        nncf_calibration_dataset = nncf.Dataset(calibration_dataloader, transform_fn)
        ov_nncf_model = nncf.quantize(ov_model, nncf_calibration_dataset,
                        preset=nncf.QuantizationPreset.MIXED,)
        ov.save_model(ov_nncf_model, ov_int8_model_path)
    else:
        ov_nncf_model = core.read_model(ov_int8_model_path)
    return ov_nncf_model

def ov_infer(ov_model, inputs, device="CPU"):
    print("=====OpenVINO backend Inference=====")
    ov_compiled_model = core.compile_model(ov_model, device)
    result = ov_compiled_model(inputs)
    return result

def get_bert_feature(text, word2ph, device=None, model_id='hfl/chinese-roberta-wwm-ext-large',
                     use_ov=True, int8_flag=False):
    if model_id not in models:
        models[model_id] = AutoModelForMaskedLM.from_pretrained(
            model_id,
            output_hidden_states=True
        ).to(device)
        tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    model = models[model_id]
    tokenizer = tokenizers[model_id]

    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)

        if use_ov:
            onnx_model_path = "ZH_bert.onnx"
            ov_model_path="ZH_bert.xml"
            ov_int8_model_path = ov_model_path.replace(".xml","_int8.xml")

            ov_input={
                "input_ids":inputs["input_ids"],
                "token_type_ids":inputs["token_type_ids"],
                "attention_mask":inputs["attention_mask"],
            }

            onnx_model_convert(onnx_model_path, model, tokenizer)
            ov_model = ov_model_convert(onnx_model_path, ov_input ,ov_model_path)
            if int8_flag:
                ov_model = ov_nncf_quant(ov_model, ov_input, ov_int8_model_path)
            res = ov_infer(ov_model, ov_input).values()
            res = list(res)
            print("=====len(res)=====",len(res),type(res),type(res[0]))
            res = np.concatenate(res[-3:-2], axis=-1)[0]
            res = torch.from_numpy(res)
        else:
            res = model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # import pdb; pdb.set_trace()
    # assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

if __name__ == "__main__":
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
