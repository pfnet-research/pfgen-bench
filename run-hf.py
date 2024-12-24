import os
import argparse
import typing
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pfgen


def callback(
    tasks: typing.List[typing.Dict[str, str]], params: typing.Dict[str, typing.Any]
) -> typing.Iterator[typing.Optional[str]]:
    model = params.get("_path", None) or params["model"]
    mode = params["mode"]
    if not hasattr(callback, "model"):
        callback.tokenizer = AutoTokenizer.from_pretrained(
            model, padding_side="left", trust_remote_code=True
        )
        callback.tokenizer.add_eos_token = False
        model_kwargs = {}
        device = params.get("_device", "cpu")
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        callback.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            **model_kwargs,
        )
        callback.model.eval()
        if device != "auto":
            callback.model.to(device)
    tokenizer = callback.tokenizer
    if not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token
    if params.get("chat_template", None):
        tokenizer.chat_template = params["chat_template"]
    model = callback.model
    if not hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.eos_token_id

    task_groups = [[]]
    for task in tasks:
        if len(task_groups[-1]) >= params["_batch_size"]:
            task_groups.append([])
        task_groups[-1].append(task)

    for task_group in task_groups:
        if mode == "completion":
            inputs = tokenizer(
                [t["prompt"] for t in task_group], return_tensors="pt", padding=True
            )
        elif mode == "chat" or mode == "qa":
            chats = []
            for task in task_group:
                if "system_prompt" in task:
                    chat = [
                        {"role": "system", "content": task["system_prompt"]},
                        {"role": "user", "content": task["user_prompt"]},
                    ]
                else:
                    chat = [{"role": "user", "content": task["prompt"]}]
                chats.append(chat)
            inputs = tokenizer.apply_chat_template(
                conversation=chats,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        try:
            outputs = model.generate(
                **{k: v.to(model.device) for k, v in inputs.items()},
                max_new_tokens=params.get("max_tokens", 300),
                do_sample=True,
                temperature=params["temperature"],
                top_p=params["top_p"],
                pad_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                stop_strings=params.get("stop", []),
            )
        except Exception as e:
            print(e)
            for _ in task_group:
                yield None
            continue
        for output in outputs:
            result = tokenizer.decode(
                output[inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            for stop in params.get("stop", []):
                if result.endswith(stop):
                    result = result[: -len(stop)]
            yield result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of trials to run."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument("--top-p", type=float, default=0.98, help="Top-p for sampling.")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for sampling."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device for sampling."
    )
    args = parser.parse_args()
    kwargs = {}
    if args.mode != "completion" and os.path.exists("chat_templates.json"):
        with open("chat_templates.json") as f:
            chat_templates = json.load(f)
            for t in chat_templates:
                if args.model in t["models"]:
                    kwargs["chat_template"] = t["chat_template"]
    pfgen.run_tasks(
        args.mode,
        callback,
        engine="hf",
        model=args.model,
        num_trials=args.num_trials,
        temperature=args.temperature,
        top_p=args.top_p,
        _path=args.path,
        _batch_size=args.batch_size,
        _device=args.device if torch.cuda.is_available() else "cpu",
        **kwargs,
    )
