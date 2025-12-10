import torch
import numpy as np
import torch.distributed as dist
from contextlib import contextmanager
import builtins
import datetime
from fairscale.nn.model_parallel import initialize as fs_init
import sys
import os
import torch.nn as nn
from torch.amp import autocast
import atexit
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            rank=0,
            world_size=1
        )
        fs_init.initialize_model_parallel(1)
        atexit.register(dist.destroy_process_group)

setup_distributed()
torch.cuda.set_device(0)

local_path = '/home/nsccgz/liopank/RAG4MTS/baselines/checkpoints/onellm_model/OneLLM'
sys.path.append(local_path)
from model.meta import MetaModel
from data.conversation_lib import conv_templates

# 设置随机种子
torch.manual_seed(1)
np.random.seed(1)

def setup_for_distributed(is_master=True):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        now = datetime.datetime.now()
        builtin_print('[{}] '.format(now), end='')
        builtin_print(*args, **kwargs)
    builtins.print = print

setup_for_distributed()

target_dtype = torch.float16
device = "cuda"

@contextmanager
def default_tensor_type(dtype=target_dtype, device=device):
    _tensor_type_stack = [(torch.float, "cpu")]

    assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
    assert dtype in [torch.float, torch.bfloat16, torch.half], "Invalid dtype"

    prev_dtype, prev_device = _tensor_type_stack[-1]
    _tensor_type_stack.append((dtype, device))
    
    try:
        torch.set_default_tensor_type(torch.empty((), dtype=dtype, device=device).type())
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        yield
    finally:
        _tensor_type_stack.pop()
        torch.set_default_tensor_type(torch.empty((), dtype=prev_dtype, device=prev_device).type())
        torch.set_default_device(prev_device)
        torch.set_default_dtype(prev_dtype)

class OnellmBackbone(nn.Module):
    def __init__(self, llama_type="onellm", llama_ckpt_dir=None, 
                 llama_config=os.path.join(local_path, "config/llama2/7B.json"),
                 tokenizer_path=os.path.join(local_path, "config/llama2/tokenizer.model")):
        super().__init__()

        with default_tensor_type(dtype=target_dtype, device=device):
            self.model = MetaModel(llama_type, llama_config, llama_ckpt_dir, tokenizer_path)

        pretrained_path = "/home/nsccgz/liopank/RAG4MTS/baselines/checkpoints/onellm_model/OneLLM/OneLLM-7B/consolidated.00-of-01.pth"
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.half().eval()

    def encode_imu(self, imu_data):
        """Encode IMU data with prompt"""
        with torch.no_grad():
            imu_feats = self.model.llma.encode_image(imu_data, modal='imu')
        return imu_feats
    
    def response_imu(self, imu_data, prompt='Describe the scene.'):

        with torch.no_grad():
            """response IMU data with prompt"""
            imu_data = imu_data.cuda()
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            prompts = [prompt_text] * imu_data.size(0)
            
            with autocast(device_type=device, dtype=target_dtype):
                responses = self.model.generate(prompts, imu_data, 64, temperature=0.1, top_p=0.75, modal=['imu'])
                
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)

        torch.cuda.empty_cache()
        return outputs

    def forward(self, imu_data):
        with torch.no_grad():
            imu_features = self.encode_imu(imu_data)
        return imu_features


class OnellmHead(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        return self.fc(x)


class OnellmModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = OnellmBackbone()

        self.upsampler = nn.Upsample(size=400, mode='linear', align_corners=True)
        self.head = OnellmHead(num_classes)

    def forward(self, imu_data, return_features=False):

        imu_data = imu_data.to(target_dtype)
        imu_data = imu_data.permute(0, 2, 1)
        imu_data = self.upsampler(imu_data)

        if return_features:
            features = self.backbone.encode_imu(imu_data)
            return features.float()
        else:
            output_imu = self.backbone.response_imu(imu_data)
            return output_imu

    def pretrain(self, x):
        """Pre-training method"""
        z = self.backbone(x, 'Describe the scene.')  # B x M x hidden_dim
        return z