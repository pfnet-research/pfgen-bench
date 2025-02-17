import glob
import hashlib
import json
import lzma
import math
import os
import re
import sys
import typing
from concurrent import futures


def generate_ngrams(text: str, n_gram: int) -> typing.Iterator[list[str]]:
    s = set()
    for p in range(1, len(text)):
        r = []
        for n in range(1, min(n_gram + 1, p)):
            t = text[p - n : p]
            if t in s:
                continue
            s.add(t)
            r.append(t)
        yield r


class NgramScorer(object):
    def __init__(self, answers, *, fluency_n_gram=10, truthfulness_n_gram=3):
        self.answers = answers
        self.fluency_n_gram = fluency_n_gram
        self.truthfulness_n_gram = truthfulness_n_gram
        self.dist = {}
        self.baseline = 1.0
        self.build()

    def build(self):
        for answer in self.answers:
            for tt in generate_ngrams(f"""^{answer}$""", self.fluency_n_gram):
                for t in tt:
                    self.dist[t] = self.dist.get(t, 0) + 1
        baseline = 0
        for answer in self.answers:
            baseline += self.score_fluency(answer)[0]
        self.baseline = baseline / len(self.answers)

    def score_fluency(self, answer):
        score = 0
        best = (0.0, 1.0)
        for n, tt in enumerate(generate_ngrams(f"""^{answer}$"""[:202], self.fluency_n_gram)):
            for t in tt:
                score += self.dist.get(t, 0)
            if n == 0:
                continue
            discount = 1 - max(n - 100, 0) / 50
            s = score * discount / self.baseline
            if s > best[0]:
                best = (s, discount)
        return best

    def score_truthfulness(self, answer):
        text = f"""^{answer}$"""[:202]
        a = [0 for _ in range(len(text))]
        for i in range(0, len(text) - self.truthfulness_n_gram + 1):
            t = text[i : i + self.truthfulness_n_gram]
            if t in self.dist:
                for j in range(i, i + self.truthfulness_n_gram):
                    a[j] = max(a[j], self.dist[t])
        total = 0
        count = 0
        score = 0.0
        best = 0.0
        for n, (c, s) in enumerate(zip(text, a)):
            if c in "^$、。・「」『』（）【】［］〈〉《》":
                continue
            total += min(1.0, s / len(self.answers) * 200)
            count += 1
            score = total / count * (1 - max(n - 100, 0) / 50)
            if n >= 100:
                best = max(best, score)
        return max(best, score)


class KeywordScorer(object):
    def __init__(self, keywords):
        self.keywords = keywords

    def match(self, answer: str, keyword: dict[str, typing.Any]) -> tuple[int, str]:
        if "t" in keyword:
            r = re.search(keyword["t"], answer)
            return r.end() if r else 9999, keyword.get("name", keyword["t"])
        if "and" in keyword:
            xs = [self.match(answer, x) for x in keyword["and"]]
            xs.sort(key=lambda x: -x[0])
            return xs[0][0], keyword.get("name", xs[0][1])
        if "or" in keyword:
            xs = [self.match(answer, x) for x in keyword["or"]]
            xs.sort(key=lambda x: x[0])
            return xs[0][0], keyword.get("name", xs[0][1])
        raise ValueError(f"Invalid keyword: {keyword}")

    def score(self, answer: str):
        results = []
        scores = [1 - max(i - 100, 0) / 50 for i in range(len(answer) + 1)]
        scores = [s for s in scores if s >= 0]
        for k in self.keywords:
            r = self.match(answer, k) + (1 - k.get("importance", 1.0),)
            for i in range(min(len(scores), r[0])):
                scores[i] *= r[2]
            results.append(r)
        n = max(reversed(range(len(scores))), key=lambda x: scores[x])
        return scores[n], [r[1:] for r in results if n < r[0]] + (
            [(f"{n - 100}字超過", 1 - max(n - 100, 0) / 50)] if n > 100 and scores[n] > 0 else []
        )


class Scorer(object):
    def __init__(self, data, *, fluency_n_gram=10, truthfulness_n_gram=3):
        self.data = data
        self.ngram_scorers = {}
        for k, v in data["answers"].items():
            self.ngram_scorers[k] = NgramScorer(
                v,
                fluency_n_gram=fluency_n_gram,
                truthfulness_n_gram=truthfulness_n_gram,
            )
        self.keyword_scorer = KeywordScorer(data["keywords"])

    def score(self, answer):
        scores = {"fluency": {}, "fluency_discount": 1.0, "truthfulness": {}}
        for k, v in self.ngram_scorers.items():
            fluency, discount = v.score_fluency(answer)
            scores["fluency"][k] = round(fluency / len(self.ngram_scorers), 6)
            scores["fluency_discount"] = round(max(scores["fluency_discount"], discount), 2)
            scores["truthfulness"][k] = round(
                v.score_truthfulness(answer) / len(self.ngram_scorers), 6
            )
        helpfulness, results = self.keyword_scorer.score(answer[:200])
        scores["helpfulness"] = round(helpfulness, 5)
        scores["helpfulness_results"] = results
        scores["average"] = round(
            (
                sum(scores["fluency"].values())
                + sum(scores["truthfulness"].values())
                + scores["helpfulness"]
            )
            / 3,
            5,
        )
        return scores


def mean_std(scores: list[float], ndigits=4) -> tuple[float, float]:
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((x - mean) ** 2 for x in scores) / len(scores))
    return round(mean, ndigits), round(std, ndigits)


class Executor(object):
    def __init__(self, input_paths: list[str]) -> None:
        super().__init__()
        self.metadata_paths = list(sorted(glob.glob("data/Q*.json")))
        self.input_paths = sum([glob.glob(x, recursive=True) for x in input_paths], [])

    def run_scorer(self, metadata, answers: dict[str, dict[str, typing.Any]]) -> None:
        print(f"""Building scorer for {metadata["question_id"]}...""", file=sys.stderr)
        scorer = Scorer(metadata)
        print(f"""Scoring {metadata["question_id"]}...""", file=sys.stderr)
        for output_path, data in answers.items():
            with open(output_path + ".tmp", "w") as f:
                for d in data["answers"]:
                    d["scores"] = scorer.score(d["answer"])
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.rename(output_path + ".tmp", output_path)
        print(f"""Finsihed scoring {metadata["question_id"]}.""", file=sys.stderr)

    def run_result(
        self,
        output_path: str,
        info: dict[str, dict[str, typing.Any]],
    ) -> None:
        print(f"""Writing result to {output_path}...""", file=sys.stderr)
        directory = os.path.dirname(output_path)
        result = {
            "input_hash": info["input_hash"],
            "metadata_hash": info["metadata_hash"],
        }
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            result["config"] = json.load(open(config_path))
        score_path = os.path.join(directory, "score.json")
        print(f"Processing {output_path}...", file=sys.stderr)
        data = {}
        for question_id, score_path in info["score_paths"].items():
            with open(score_path, "rt") as f:
                data[question_id] = json.load(f)["answers"]
        data = dict(sorted(data.items(), key=lambda x: x[0]))
        with open(score_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        result["num_trials"] = min([len(x) for x in data.values()])
        scores = [[a["scores"]["average"] for a in x] for x in data.values()]
        scores = [sum(x) / len(x) for x in zip(*scores)]
        result["score"], result["score_std"] = mean_std(scores)
        lengths = sum([[len(a["answer"]) for a in x] for x in data.values()], [])
        result["length"], result["length_std"] = mean_std(lengths, 1)

        result_questions = {}
        scores_all = {"fluency": {}, "truthfulness": {}}
        for question_id, answers in data.items():
            answers.sort(key=lambda x: x["scores"]["average"], reverse=True)

            scores = {"fluency": {}, "truthfulness": {}}
            for m in ["fluency", "truthfulness"]:
                for k in answers[0]["scores"][m]:
                    scores[m][k] = round(
                        sum([a["scores"][m][k] for a in answers]) / len(answers), 5
                    )
                    scores_all[m][k] = round(
                        scores_all[m].get(k, 0.0) + scores[m][k] / len(data), 5
                    )
            for m in ["helpfulness", "average"]:
                scores[m] = round(sum([a["scores"][m] for a in answers]) / len(answers), 5)
                scores_all[m] = round(scores_all.get(m, 0.0) + scores[m] / len(data), 5)

            samples = []
            index_seen = set()
            for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
                index = round(ratio * (len(answers) - 1))
                if index in index_seen:
                    continue
                index_seen.add(index)
                a = answers[index].copy()
                del a["question"]
                samples.append(a)

            r = {"question": answers[0]["question"]}
            r["score"], r["score_std"] = mean_std([a["scores"]["average"] for a in answers])
            r["length"], r["length_std"] = mean_std([len(a["answer"]) for a in answers], 1)
            r["scores"] = scores
            r["samples"] = samples
            result_questions[question_id] = r
        result["scores"] = scores_all
        result["questions"] = result_questions

        result_path = os.path.join(directory, "result.json")
        with open(result_path + ".tmp", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        os.rename(result_path + ".tmp", result_path)

    def run(self, force=False):
        print("Loading metadata...", file=sys.stderr)
        metadata = {}
        metadata_hash = {}
        for metadata_path in self.metadata_paths:
            d = json.load(open(metadata_path))
            metadata[d["question"]] = d
            # Calculate SHA-1 hash.
            with open(metadata_path, "rb") as f:
                metadata_hash[d["question"]] = hashlib.sha1(f.read()).hexdigest()
        metadata_all_hash = hashlib.sha1(
            "".join(metadata_hash.values()).encode("utf-8")
        ).hexdigest()

        print("Loading answers...", file=sys.stderr)
        answers = {}
        result_info = {}
        for input_path in self.input_paths:
            # Calculate SHA-1 hash.
            with open(input_path, "rb") as f:
                input_hash = hashlib.sha1(f.read()).hexdigest()

            # Check if the result needs to be updated.
            result_path = os.path.join(os.path.dirname(input_path), "result.json")
            if not force and os.path.exists(result_path):
                with open(result_path, "rt") as f:
                    d = f.read()
                    if input_hash in d and metadata_all_hash in d:
                        continue

            # Register the result path.
            result_info[result_path] = {
                "input_path": input_path,
                "score_paths": {},
                "input_hash": input_hash,
                "metadata_hash": metadata_all_hash,
            }

            with (
                lzma.open(input_path, "rt")
                if input_path.endswith(".xz")
                else open(input_path, "rt")
            ) as f:
                for line in f:
                    d = json.loads(line)
                    assert (
                        d["question"] in metadata
                    ), f"""Question {d["question"]} not found in metadata."""

                    question_id = metadata[d["question"]]["question_id"]
                    output_path = os.path.join(
                        os.path.dirname(input_path),
                        "cache",
                        f"""score_{question_id}.json""",
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    result_info[result_path]["score_paths"][question_id] = output_path

                    answers[d["question"]] = answers.get(d["question"], {})
                    answers[d["question"]][output_path] = answers[d["question"]].get(
                        output_path,
                        {
                            "input_hash": input_hash,
                            "metadata_hash": metadata_hash[d["question"]],
                            "answers": [],
                        },
                    )
                    answers[d["question"]][output_path]["answers"].append(d)
            # Check if the output needs to be updated.
            if not force:
                for question, data in answers.items():
                    for output_path in list(data.keys()):
                        if not os.path.exists(output_path):
                            continue
                        with open(output_path, "rt") as f:
                            output = f.read()
                            if input_hash not in output:
                                continue
                            if metadata_hash[question] not in output:
                                continue
                        del answers[question][output_path]

        print("Scoring...", file=sys.stderr)
        with futures.ProcessPoolExecutor(max_workers=50) as executor:
            fs = []
            for m in metadata.values():
                if m["question"] not in answers:
                    continue
                fs.append(executor.submit(self.run_scorer, m, answers[m["question"]]))
            for f in futures.as_completed(fs):
                f.result()

        print("Writing results...", file=sys.stderr)
        with futures.ProcessPoolExecutor(max_workers=50) as executor:
            fs = []
            for result_path, info in result_info.items():
                fs.append(executor.submit(self.run_result, result_path, info))
            for f in futures.as_completed(fs):
                f.result()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--force", action="store_true")
    argparser.add_argument("input", type=str, nargs="+")
    args = argparser.parse_args()
    Executor(args.input).run(force=args.force)
