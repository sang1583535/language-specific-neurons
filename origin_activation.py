import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams

import os
os.environ["VLLM_USE_V1"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-l", "--lang", type=str, default="zh")
args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
is_olmo = bool(args.model.lower().find('olmo') >= 0)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
if is_olmo:
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
elif is_llama:
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
else:
    intermediate_size = model.llm_engine.model_config.hf_config.hidden_size * 4

over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
        activation = gate_up[:, :, : i // 2].float() # b, l, i
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def olmo_forward(self, x: torch.Tensor):
        # OLMo uses ff_proj -> activation (swiglu) -> ff_out
        x = self.ff_proj(x)
        # swiglu activation: split, apply silu to first half, then multiply
        i = x.size(-1)
        x_gate = x[:, :, : i // 2]
        x_value = x[:, :, i // 2 :]
        x_gate = torch.nn.SiLU()(x_gate)
        activation = x_gate.float()  # b, l, i
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = x_gate * x_value
        x = self.ff_out(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    elif is_olmo:
        return olmo_forward
    else:
        return bloom_forward

for i in range(num_layers):
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    elif is_olmo:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.blocks[i]
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

lang = args.lang
if is_llama:
    ids = torch.load(f'data/id.{lang}.train.llama')
elif is_olmo:
    ids = torch.load(f'data/id.{lang}.train.olmo')
else:
    ids = torch.load(f'data/id.{lang}.train.bloom')
l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

output = dict(n=l, over_zero=over_zero.to('cpu'))

if is_llama:
    torch.save(output, f'data/activation.{lang}.train.llama-7b')
elif is_olmo:
    torch.save(output, f'data/activation.{lang}.train.olmo')
else:
    torch.save(output, f'data/activation.{lang}.train.bloom-7b')
