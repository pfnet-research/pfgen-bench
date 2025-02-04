import os
import argparse
import typing

import openai

import pfgen


def callback(
    tasks: typing.List[typing.Dict[str, str]], params: typing.Dict[str, typing.Any]
) -> typing.Iterator[typing.Optional[str]]:
    model = params["model"].split("/")[-1]
    mode = params["mode"]
    temperature = params["temperature"]
    kwargs = {}
    kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    client = openai.OpenAI(
        **kwargs
    )
    for task in tasks:
        kwargs = {}
        if mode == "chat":
            kwargs["messages"] = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_prompt"]},
            ]
        elif mode == "qa":
            kwargs["messages"] = [{"role": "user", "content": task["prompt"]}]
        elif mode == "completion":
            kwargs["prompt"] = task["prompt"]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        try:
            if mode in ["qa", "chat"]:
                results = client.chat.completions.create(
                    model=params["model"],
                    max_tokens=params.get("max_tokens", 500),
                    temperature=temperature,
                    stop=params.get("stop", []),
                    **kwargs,
                )
                yield results.choices[0].message.content.removeprefix("A:").strip()
            elif mode == "completion":
                results = client.completions.create(
                    model=params["model"],
                    max_tokens=params.get("max_tokens", 500),
                    temperature=temperature,
                    stop=params.get("stop", []),
                    **kwargs,
                )
                yield results.choices[0].text.strip()
        except openai.OpenAIError as e:
            print(f"API Error: {e}")
            yield None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of trials to run."
    )
    args = parser.parse_args()
    pfgen.run_tasks(
        args.mode,
        callback,
        engine="openai-api",
        model=args.model,
        temperature=args.temperature,
        num_trials=args.num_trials,
    )
