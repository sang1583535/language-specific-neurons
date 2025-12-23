import os
# Ensure vLLM uses the legacy engine layout where model internals are reachable.
# Must be set BEFORE importing vllm.
os.environ.setdefault("VLLM_USE_V1", "0")

import argparse
from types import MethodType

import torch
import torch.nn.functional as F
import torch.distributed as dist

from vllm import LLM, SamplingParams


def get_vllm_model(llm: LLM):
    """
    Get the underlying PyTorch model from vLLM 0.10.x in V0 mode.
    """
    eng = llm.llm_engine
    if not hasattr(eng, "model_executor"):
        raise RuntimeError(
            "Cannot access model_executor. You may be on vLLM v1 engine.\n"
            "Set VLLM_USE_V1=0 BEFORE importing vllm."
        )
    # driver_worker exists under model_executor in V0 engine
    return eng.model_executor.driver_worker.model_runner.model


def get_layers_container(vllm_model):
    """
    Locate the decoder layers container across common vLLM model wrappers.
    Returns a list/ModuleList of layers.
    """
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "layers"),
        ("transformer", "h"),
        ("transformer", "blocks"),
    ]
    for path in candidate_paths:
        obj = vllm_model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return obj
    raise RuntimeError(
        "Cannot locate decoder layers on the vLLM model. "
        "Try printing the model structure to find the correct path."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-l", "--lang", type=str, default="zh")
    parser.add_argument("--max_tokens_to_use", type=int, default=100_000_000,
                        help="Cap total tokens used from the id.* tensor before reshaping.")
    parser.add_argument("--data_prefix", type=str, default="data",
                        help="Folder containing id.* token tensors and where outputs will be saved.")
    parser.add_argument("--no_reduce", action="store_true",
                        help="Do not all_reduce over tensor-parallel ranks (debug).")
    args = parser.parse_args()

    # mkdir data_prefix if not exists
    os.makedirs(args.data_prefix, exist_ok=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
    )

    vllm_model = get_vllm_model(llm)

    hf_config = llm.llm_engine.model_config.hf_config
    model_type = getattr(hf_config, "model_type", "")
    model_type_l = str(model_type).lower()
    model_name_l = args.model.lower()

    is_llama = ("llama" in model_type_l) or ("llama" in model_name_l)
    is_olmo2 = ("olmo2" in model_type_l) or ("olmo-2" in model_name_l) or ("olmo2" in model_name_l)

    max_length = llm.llm_engine.model_config.max_model_len
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    intermediate_size = int(getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4))

    # Counts stored on GPU for speed
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32, device="cuda")

    layers = get_layers_container(vllm_model)

    def make_swiglu_like_forward(layer_idx: int):
        """
        Works for both LLaMA-style and vLLM OLMo2 MLP, as both expose:
          - self.gate_up_proj
          - self.down_proj
        with gate_up_proj producing last-dim = 2*intermediate_size.
        """
        def forward(self, x: torch.Tensor):
            # x is usually [T, H] in vLLM; sometimes could be [B, L, H]
            gate_up, _ = self.gate_up_proj(x)  # [T, 2I] or [B, L, 2I]
            last = gate_up.size(-1)
            i = last // 2

            if gate_up.dim() == 2:
                # [T, 2I]
                gate = gate_up[:, :i]
                up = gate_up[:, i:]
                gate_act = F.silu(gate)  # [T, I]
                over_zero[layer_idx, :] += (gate_act > 0).sum(dim=0).to(torch.int32)
                out = gate_act * up  # [T, I]
            elif gate_up.dim() == 3:
                # [B, L, 2I]
                gate = gate_up[:, :, :i]
                up = gate_up[:, :, i:]
                gate_act = F.silu(gate)  # [B, L, I]
                over_zero[layer_idx, :] += (gate_act > 0).sum(dim=(0, 1)).to(torch.int32)
                out = gate_act * up  # [B, L, I]
            else:
                raise RuntimeError(f"Unexpected gate_up dim={gate_up.dim()} shape={tuple(gate_up.shape)}")

            out, _ = self.down_proj(out)
            return out

        return forward

    # Patch each layer MLP forward
    for i in range(num_layers):
        layer = layers[i]
        if not hasattr(layer, "mlp"):
            raise RuntimeError(f"Layer {i} has no .mlp; got {type(layer)}")
        mlp = layer.mlp
        # Ensure required members exist
        if not hasattr(mlp, "gate_up_proj") or not hasattr(mlp, "down_proj"):
            raise RuntimeError(
                f"Layer {i} mlp missing gate_up_proj/down_proj. type={type(mlp)}. "
                "This script currently supports SwiGLU MLPs (LLaMA/OLMo2-style)."
            )
        mlp.forward = MethodType(make_swiglu_like_forward(i), mlp)

    # Load token IDs
    lang = args.lang
    if is_llama:
        ids_path = f"{args.data_prefix}/id.{lang}.train.llama"
        out_suffix = "llama"
    elif is_olmo2:
        ids_path = f"{args.data_prefix}/id.{lang}.train.olmo2"
        out_suffix = "olmo2"
    else:
        ids_path = f"{args.data_prefix}/id.{lang}.train.other"
        out_suffix = "other"

    ids = torch.load(ids_path)  # expected 1D LongTensor
    if not isinstance(ids, torch.Tensor):
        raise RuntimeError(f"Loaded ids is not a torch.Tensor from {ids_path}")

    total = int(ids.size(0))
    # Cap tokens then trim to a multiple of max_length
    use = min(total, int(args.max_tokens_to_use))
    use = (use // max_length) * max_length
    ids = ids[:use]

    # Reshape into prompts of fixed length
    input_ids = ids.reshape(-1, max_length)  # [N, max_length]

    # Wrap each row as TokensPrompt
    prompts = [{"prompt_token_ids": row} for row in input_ids.tolist()]

    # Run inference to trigger patched forwards
    _ = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(max_tokens=1),
        use_tqdm=True,
    )

    # Reduce counts across tensor-parallel ranks if distributed is initialized
    if (not args.no_reduce) and dist.is_available() and dist.is_initialized():
        dist.all_reduce(over_zero, op=dist.ReduceOp.SUM)

    output = {
        "n": use,
        "over_zero": over_zero.cpu(),
        "model": args.model,
        "model_type": model_type,
        "max_length": max_length,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
    }

    out_path = f"{args.data_prefix}/activation.{lang}.train.{out_suffix}"
    torch.save(output, out_path)

    print(f"Saved: {out_path}")
    print(f"Used tokens: {use} (max_length={max_length}, prompts={input_ids.size(0)})")


if __name__ == "__main__":
    main()
