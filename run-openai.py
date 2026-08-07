import argparse
import os
import typing
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import openai

import pfgen


def generate(
    client: openai.OpenAI,
    task: dict[str, str],
    params: dict[str, typing.Any],
    stop: list[str],
    add_no_think: bool,
) -> str | None:
    mode = params["mode"]
    kwargs: dict[str, typing.Any] = {}
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
        if mode in ["qa", "chat"]:
            results = client.chat.completions.create(
                model=params["model"],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                stop=stop,
                **kwargs,
            )
            return results.choices[0].message.content.removeprefix("A:").strip()
        else:
            results = client.completions.create(
                model=params["model"],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                stop=stop,
                stream=False,
                **kwargs,
            )
            return results.choices[0].text.strip()
    except openai.OpenAIError as e:
        print(f"API Error: {e}")
        return None


def callback(
    tasks: list[dict[str, str]],
    params: dict[str, typing.Any],
    extra_eos_tokens: list[str] | None,
    add_no_think: bool,
    batch_size: int,
) -> typing.Iterator[str | None]:
    kwargs: dict[str, typing.Any] = {}
    kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(**kwargs)
    stop = list(params.get("stop", []))
    if extra_eos_tokens is not None:
        stop = list(set(stop + extra_eos_tokens))

    if batch_size <= 1:
        for task in tasks:
            yield generate(client, task, params, stop, add_no_think)
        return

    # Send `batch_size` requests concurrently so that the server can batch them.
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for begin in range(0, len(tasks), batch_size):
            futures = [
                executor.submit(generate, client, task, params, stop, add_no_think)
                for task in tasks[begin : begin + batch_size]
            ]
            for future in futures:
                yield future.result()


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of requests sent concurrently.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate (overrides default).",
    )
    parser.add_argument("--extra-eos-tokens", type=str, nargs="+", help="Extra EOS strings")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable reasoning when generation by Qwen3 models",
    )
    parser.add_argument("--num-retries", type=int, default=10, help="Number of retries.")
    parser.add_argument(
        "--ignore-failure",
        action="store_true",
        default=False,
        help="Do not throw an exception if answer generation fails.",
    )
    args = parser.parse_args()

    wrapped_callback = partial(
        callback,
        extra_eos_tokens=args.extra_eos_tokens,
        add_no_think=args.disable_thinking,
        batch_size=args.batch_size,
    )

    # Prepare optional kwargs
    extra_kwargs = {}
    if args.max_tokens is not None:
        extra_kwargs["max_tokens"] = args.max_tokens

    pfgen.run_tasks(
        args.mode,
        wrapped_callback,
        engine="openai-api",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_trials=args.num_trials,
        enable_thinking=not args.disable_thinking,
        num_retries=args.num_retries,
        ignore_failure=args.ignore_failure,
        **extra_kwargs,
    )
