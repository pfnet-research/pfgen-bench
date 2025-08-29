import argparse
import typing

import pfgen


def callback(
    tasks: list[dict[str, str]], params: dict[str, typing.Any]
) -> typing.Iterator[str | None]:
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
    parser.add_argument("--num-retries", type=int, default=10, help="Number of retries.")
    parser.add_argument(
        "--ignore-failure",
        action="store_true",
        default=False,
        help="Do not throw an exception if answer generation fails.",
    )
    args = parser.parse_args()
    pfgen.run_tasks(
        args.mode,
        callback,
        engine="manual",
        model=args.model,
        num_retries=args.num_retries,
        ignore_failure=args.ignore_failure,
    )
