import argparse
import json
import math
import os
import typing

import torch
import transformers
from transformers import AutoConfig
from vllm import LLM, SamplingParams

import pfgen


class Callback:
    def __init__(self) -> None:
        self.llm: LLM | None = None
        self.tokenizer: transformers.PreTrainedTokenizer | None = None

    def __call__(
        self, tasks: list[dict[str, str]], params: dict[str, typing.Any]
    ) -> typing.Iterator[str | None]:
        model = params.get("_path", None) or params["model"]
        mode = params["mode"]
        if self.llm is None:
            kwargs = {}
            if "dtype" in params:
                kwargs["dtype"] = params["dtype"]
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            model_max_tokens = getattr(config, "max_position_embeddings", None)
            model_seq_length = getattr(config, "seq_length", None)
            if "_max_tokens" in params:
                m = params["_max_tokens"]
                kwargs["max_num_batched_tokens"] = m
                kwargs["max_model_len"] = min(m, model_max_tokens or m, model_seq_length or m)
            tensor_parallel_size = math.gcd(
                torch.cuda.device_count(),
                math.gcd(
                    getattr(config, "num_attention_heads", 720720),
                    getattr(config, "num_key_value_heads", 720720),
                ),
            )
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                **kwargs,
            )
            # Workaround for multiple vLLM versions.
            self.tokenizer = getattr(self.llm, "tokenizer", self.llm).get_tokenizer()  # type: ignore
            if hasattr(self.tokenizer, "add_eos_token"):
                self.tokenizer.add_eos_token = False
        assert self.llm is not None
        llm = self.llm
        assert self.tokenizer is not None
        tokenizer = self.tokenizer

        stop = params.get("stop", []).copy()
        if tokenizer.eos_token is not None:
            stop.append(tokenizer.eos_token)
        if tokenizer.bos_token is not None:
            stop.append(tokenizer.bos_token)
        sampling_params = SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params.get("max_tokens", 300),
            skip_special_tokens=False,
            stop=stop,
            seed=tasks[0].get("seed", None),
        )
        if mode == "completion":
            for t in tasks:
                t["prompt_token_ids"] = tokenizer.encode(t["prompt"])

            # Workaround for tokenizers adding an EOS token (e.g.,
            # sbintuitions/sarashina1-7b).
            # NOTE: This is necessary only for completion because chat templates bypasses
            # adding special tokens.
            if not hasattr(tokenizer, "add_eos_token") and hasattr(tokenizer, "eos_token_id"):
                for t in tasks:
                    if (
                        len(t["prompt_token_ids"]) > 1
                        and t["prompt_token_ids"][-1] == tokenizer.eos_token_id
                    ):
                        t["prompt_token_ids"] = t["prompt_token_ids"][:-1]
        else:
            if params.get("chat_template", None):
                tokenizer.chat_template = params["chat_template"]
            if mode == "chat":
                for t in tasks:
                    t["prompt_token_ids"] = tokenizer.apply_chat_template(
                        conversation=[
                            {"role": "system", "content": t["system_prompt"]},
                            {"role": "user", "content": t["user_prompt"]},
                        ],
                        add_generation_prompt=True,
                    )
            elif mode == "qa":
                for t in tasks:
                    t["prompt_token_ids"] = tokenizer.apply_chat_template(
                        conversation=[{"role": "user", "content": t["prompt"]}],
                        add_generation_prompt=True,
                    )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        outputs = llm.generate(
            prompt_token_ids=[task["prompt_token_ids"] for task in tasks],
            sampling_params=sampling_params,
        )
        for output in outputs:
            try:
                yield output.outputs[0].text.strip()
            except Exception as e:
                print(f"Error: {e}")
                yield None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mode",
        type=str,
        default="completion",
        choices=["chat", "qa", "completion"],
        help="Which chat template to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Huggingface model name.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the model.",
    )
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top-p", type=float, default=0.98, help="Top-p for sampling.")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Maximum number of tokens.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for the prompt.")
    parser.add_argument("--dtype", type=str, default="", help="Data type.")
    args = parser.parse_args()
    kwargs = {}
    if args.max_tokens:
        kwargs["_max_tokens"] = args.max_tokens
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.prefix:
        kwargs["prefix"] = args.prefix
    if args.mode != "completion" and os.path.exists("chat_templates.json"):
        with open("chat_templates.json") as f:
            chat_templates = json.load(f)
            for t in chat_templates:
                if args.model in t["models"]:
                    kwargs["chat_template"] = t["chat_template"]
    pfgen.run_tasks(
        args.mode,
        Callback(),
        engine="vllm",
        model=args.model,
        num_examples=args.num_examples,
        num_trials=args.num_trials,
        temperature=args.temperature,
        top_p=args.top_p,
        _path=args.path,
        **kwargs,
    )
