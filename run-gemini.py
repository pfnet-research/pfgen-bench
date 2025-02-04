import os
import argparse
import typing
import time

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

import pfgen


def callback(
    tasks: typing.List[typing.Dict[str, str]], params: typing.Dict[str, typing.Any]
) -> typing.Iterator[typing.Optional[str]]:
    model = params["model"].split("/")[-1]
    assert params["mode"] == "qa"
    project = os.environ.get("VERTEXAI_PROJECT", "")
    location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
    assert project and location
    vertexai.init(project=project, location=location)
    model = GenerativeModel(model)
    for task in tasks:
        for trial in range(10):
            try:
                responses = model.generate_content(
                    [task["prompt"]],
                    generation_config={
                        "max_output_tokens": params.get("max_tokens", 500),
                        "temperature": params.get("temperature", 1.0),
                        "top_p": params.get("top_p", 1.0),
                    },
                    safety_settings=[
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                    ],
                )
                if params.get("multi_choice", False):
                    yield responses.candidates[0].content.parts[-1].text
                else:
                    yield responses.text
            except Exception as e:
                print(f"API Error: {e}")
                if trial < 5 and f"{e}".startswith("429"):
                    print("Rate limited, retrying after 20 seconds...")
                    time.sleep(20)
                    continue
                yield None
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="qa",
        choices=["chat", "qa"],
        help="Which chat template to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash-001",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--multi-choice",
        action="store_true",
        help="Use multi-choice generation.",
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
        engine="gemini",
        model="google/" + args.model.split("/")[-1],
        multi_choice=args.multi_choice,
        temperature=args.temperature,
        num_trials=args.num_trials,
        max_tokens=3000,
    )
