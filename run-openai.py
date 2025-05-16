import argparse
import os
import typing
from functools import partial

import openai

import pfgen


def callback(
    tasks: list[dict[str, str]],
    params: dict[str, typing.Any],
    extra_eos_tokens: list[str] | None,
    add_no_think: bool,
) -> typing.Iterator[str | None]:
    mode = params["mode"]
    temperature = params["temperature"]
    kwargs: dict[str, typing.Any] = {}
    kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(**kwargs)
    for task in tasks:
        kwargs = {}
        if mode == "chat":
            system_prompt = "/no_think\n" if add_no_think else ""
            system_prompt += task["system_prompt"]
            kwargs["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["user_prompt"]},
            ]
        elif mode == "qa":
            prompt = "/no_think\n" if add_no_think else ""
            prompt += task["prompt"]
            kwargs["messages"] = [{"role": "user", "content": prompt}]
        elif mode == "completion":
            kwargs["prompt"] = "/no_think\n" if add_no_think else ""
            kwargs["prompt"] += task["prompt"]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        try:
            stop = params.get("stop", [])
            if extra_eos_tokens is not None:
                stop.extend(extra_eos_tokens)
                stop = list(set(stop))
            if mode in ["qa", "chat"]:
                results = client.chat.completions.create(
                    model=params["model"],
                    max_tokens=params.get("max_tokens", 500),
                    temperature=temperature,
                    top_p=params["top_p"],
                    stop=stop,
                    **kwargs,
                )
                yield results.choices[0].message.content.removeprefix("A:").strip()
            elif mode == "completion":
                results = client.completions.create(
                    model=params["model"],
                    max_tokens=params.get("max_tokens", 500),
                    temperature=temperature,
                    top_p=params["top_p"],
                    stop=stop,
                    stream=False,
                    **kwargs,
                )
                yield results.choices[0].text.strip()
        except openai.OpenAIError as e:
            print(f"API Error: {e}")
            yield None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mode",
        type=str,
        default="qa",
        choices=["chat", "qa", "completion"],
        help="Which chat template to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="OpenAI model name.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--top-p", type=float, default=0.98, help="Top-p for sampling.")
    parser.add_argument("--extra-eos-tokens", type=str, nargs="+", help="Extra EOS strings")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable reasoning when generation by Qwen3 models",
    )
    args = parser.parse_args()

    wrapped_callback = partial(
        callback,
        extra_eos_tokens=args.extra_eos_tokens,
        add_no_think=args.disable_thinking,
    )

    pfgen.run_tasks(
        args.mode,
        wrapped_callback,
        engine="openai-api",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_trials=args.num_trials,
        enable_thinking=not args.disable_thinking,
    )
