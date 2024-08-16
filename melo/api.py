import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
import openvino as ov
from tqdm import tqdm
import torch
import nncf

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.onnx import FeaturesManager
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

core = ov.Core()

class RefactorTTSModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    noise_scale=0.6,
                    length_scale=1.0,
                    noise_scale_w=0.8,
                    sdp_ratio=0.2,):
        return self.model.infer(x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale)

class CustomDataset(Dataset):
    def __init__(self, data_count=100, dummy_data=None):
        self.dataset = []
        for i in range(data_count):
            self.dataset.append([dummy_data])
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        image = self.dataset[idx]
        return image

def transform_fn(data_item):
    data_item = data_item[0] 
    inputs = {"phones":data_item["phones"].squeeze(0),
              "phones_length":data_item["phones_length"].squeeze(0),
              "speakers":data_item["speakers"].squeeze(0),
              "tones":data_item["tones"].squeeze(0),    
              "lang_ids":data_item["lang_ids"].squeeze(0),    
              "bert":data_item["bert"].squeeze(0),         
              "ja_bert":data_item["ja_bert"].squeeze(0),    
              "sdp_ratio":data_item["sdp_ratio"].squeeze(0),    
              "noise_scale":data_item["noise_scale"].squeeze(0),    
              "noise_scale_w":data_item["noise_scale_w"].squeeze(0),   
              "length_scale":data_item["length_scale"].squeeze(0),   
            }
    return inputs

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None,
                use_ov=True,
                int8_flag=False):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
        self.use_ov = use_ov
        self.int8_flag = int8_flag
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts


    def onnx_model_convert(self,torch_model, dummy_model_input, onnx_model_path):
        print("======Onnx TTS Model Convert=======")
        if not os.path.exists(onnx_model_path):
            torch.onnx.export(
                torch_model, 
                tuple(dummy_model_input.values()),
                onnx_model_path,  
                input_names=["phones", "phones_length", "speakers",
                             "tones", "lang_ids","bert","ja_bert",
                             "sdp_ratio","noise_scale","noise_scale_w","length_scale"], 
                # output_names=['logits'], 
                dynamic_axes={'phones': {0: 'batch_size', 1: 'sequence'}, 
                              'tones': {0: 'batch_size', 1: 'sequence'}, 
                              'lang_ids': {0: 'batch_size', 1: 'sequence'}, 
                              'bert': {0: 'batch_size', 2: 'sequence'}, 
                              'ja_bert': {0: 'batch_size', 2: 'sequence'}}, 
                # do_constant_folding=True, 
                opset_version=13, 
            )

    def ov_model_convert(self,model, sample_input,ov_model_path="ZH_tts.xml"):
        print("=====OpenVINO TTS Model Convert=====")
        if not os.path.exists(ov_model_path):
            ov_model = ov.convert_model(model,example_input=sample_input)
            ov.save_model(ov_model, ov_model_path)
            print(f"== OV {ov_model_path} convert success ==")
        else:
            ov_model = core.read_model(ov_model_path)
        return ov_model

    def ov_nncf_quant(self,ov_model, sample_input, ov_int8_model_path):
        print("=====OpenVINO NNCF TTS Model Quantization=====")
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

    def ov_infer(self, ov_model, inputs, device="CPU"):
        print("=====OpenVINO backend TTS Inference=====")
        ov_compiled_model = core.compile_model(ov_model, device)
        result = ov_compiled_model(inputs)
        return result

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                if self.use_ov :
                    print("======OpenVINO convert TTS model=======")
                    x_tst = phones.unsqueeze(0)
                    # if self.int8_flag: 
                    #     speed = speed*0.3
                    #     noise_scale = noise_scale*0.2
                    #     noise_scale_w = noise_scale_w*0.2
                    #     sdp_ratio = sdp_ratio*2
                    ov_input = {
                        "phones":x_tst,
                        "phones_length":torch.LongTensor([phones.size(0)]),
                        "speakers":torch.LongTensor([speaker_id]),
                        "tones":tones.unsqueeze(0),
                        "lang_ids":lang_ids.unsqueeze(0),
                        "bert":bert.unsqueeze(0),
                        "ja_bert":ja_bert.unsqueeze(0),
                        "noise_scale":noise_scale,
                        "length_scale":1./speed,
                        "noise_scale_w":noise_scale_w,
                        "sdp_ratio":sdp_ratio,
                    }
                    onnx_model_path = "ZH_tts.onnx"
                    ov_model_path="ZH_tts.xml"
                    ov_int8_model_path = "ZH_tts_int8.xml"
                    ref_tts_model = RefactorTTSModel(self.model)

                    self.onnx_model_convert(ref_tts_model, ov_input, onnx_model_path)
                    ov_model = self.ov_model_convert(onnx_model_path, ov_input, ov_model_path)
                    if self.int8_flag:
                        ov_model = self.ov_nncf_quant(ov_model, ov_input, ov_int8_model_path)
                    res = self.ov_infer(ov_model, ov_input).values()
                    res = list(res)
                    audio = res[0][0, 0]
                else:
                    x_tst = phones.to(device).unsqueeze(0)
                    tones = tones.to(device).unsqueeze(0)
                    lang_ids = lang_ids.to(device).unsqueeze(0)
                    bert = bert.to(device).unsqueeze(0)
                    ja_bert = ja_bert.to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                    del phones
                    speakers = torch.LongTensor([speaker_id]).to(device)
                    audio = self.model.infer(
                            x_tst,
                            x_tst_lengths,
                            speakers,
                            tones,
                            lang_ids,
                            bert,
                            ja_bert,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=1. / speed,
                        )[0][0, 0].data.cpu().float().numpy()
                    del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
