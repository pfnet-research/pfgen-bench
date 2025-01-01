import argparse
import typing

import pfgen


def callback(
    tasks: typing.List[typing.Dict[str, str]], params: typing.Dict[str, typing.Any]
) -> typing.Iterator[typing.Optional[str]]:
    mode = params["mode"]
    assert mode == "qa" or mode == "completion"
    for task in tasks:
        print("=" * 80)
        print(task["prompt"])
        print()
        output = input("Enter the output: ").strip()
        if output:
            yield output
        else:
            yield None
        print()

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
    args = parser.parse_args()
    pfgen.run_tasks(
        args.mode,
        callback,
        engine="manual",
        model=args.model,
    )
