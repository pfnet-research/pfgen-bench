# pip install --upgrade google-genai pydantic

import argparse
import os
import time
import typing
from typing import Literal

from google import genai
from google.genai import types
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

import pfgen
import pydantic

# プロジェクト名等は保存する必要がなく、人によっては見せない方が良い場合もあるので環境変数に
project = os.environ.get("VERTEXAI_PROJECT", "")
location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
assert project and location

class GeminiGoogleSupportedParams(pydantic.BaseModel):
    mode: Literal["qa"] = "qa"
    model_request_name: str
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    max_tokens: int


class GeminiGoogleSupportedTask(pydantic.BaseModel):
    prompt: str


def callback(
    tasks: list[dict[str, str]], params: dict[str, typing.Any]
) -> typing.Iterator[str | None]:
    param_model = GeminiGoogleSupportedParams(**params)
    client = genai.Client(vertexai=True, project=project, location=location)

    for original_task in tasks:
        task = GeminiGoogleSupportedTask(**original_task)
        for retry_count in range(10):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=task.prompt)
                        ],
                    )
                ]
                responses = client.models.generate_content(
                    model = param_model.model_request_name,
                    contents = contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=param_model.max_tokens,
                        temperature=param_model.temperature,
                        top_p=param_model.top_p,
                        safety_settings=[
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            ),
                        ],
                        # とりあえず0にしているが、ONにするべき？基準が不明
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0
                        )
                    ),
                )
                # multi_choiceはどのようなときに使うのか
                #if params.get("multi_choice", False):
                    #yield responses.candidates[0].content.parts[-1].text
                #else:
                yield responses.text
            except Exception as e:
                print(f"API Error: {e}")
                if retry_count < 5 and f"{e}".startswith("429"):
                    print("Rate limited, retrying after 20 seconds...")
                    time.sleep(20)
                    continue
                yield None
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model-save-name",
        type=str,
        default="google/gemini-2.5-flash-preview-05-20",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--model-request-name",
        type=str,
        default="gemini-2.5-flash-preview-05-20",
        help="model name for request parameter",
    )
    """
    parser.add_argument(
        "--multi-choice",
        action="store_true",
        help="Use multi-choice generation.",
    )
    """
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials to run.")
    args = parser.parse_args()
    pfgen.run_tasks(
        "qa",
        callback,
        engine="gemini",
        model=args.model_save_name,
        temperature=args.temperature,
        num_trials=args.num_trials,
        max_tokens=3000,
        model_request_name=args.model_request_name,
    )
