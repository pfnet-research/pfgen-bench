import glob
import hashlib
import html
import json
import os
import re
import sys
from concurrent import futures


class PgenReporter(object):
    def __init__(self):
        super().__init__()
        print("Loading metadata...", file=sys.stderr)
        metadata = {}
        for metadata_path in sorted(glob.glob("data/Q*.json")):
            with open(metadata_path) as f:
                d = json.load(f)
                metadata[d["question"]] = d
        self.metadata = metadata

        print("Listing result paths...", file=sys.stderr)
        self.result_paths = list(
            sorted(
                glob.glob("result/**/result.json", recursive=True)
                + glob.glob("data/**/result.json", recursive=True)
            )
        )

    def stringfy_items(self, d: dict[str, float], *, long: bool = False):
        config = {
            "P": (0, "PFN"),
            "T": (1, "Titech"),
            "S": (2, "Stockmark"),
            "R": (3, "RAG"),
            "X": (4, "Unknown"),
        }
        d = dict(sorted(d.items(), key=lambda x: config[x[0]][0]))
        if long:
            d = {config[k][1]: v for k, v in d.items()}
        return "+".join([f"{k}:{v:.4f}" for k, v in d.items()])

    def stringfy_scores(self, scores, *, long: bool = False, extra: str = ""):
        if long:
            result = f"""- Score: {scores["average"]:.3f}{extra}\n"""
            for m, name in [("fluency", "Fluency"), ("truthfulness", "Truthfulness")]:
                score_sum = sum(scores[m].values())
                details = self.stringfy_items(scores[m], long=True)
                result += f"""  - {name}: {score_sum:.3f} ({details})\n"""
            result += f"""  - Helpfulness: {scores["helpfulness"]:.3f}\n"""
        else:
            result = f"""{scores["average"]:.3f} (=avg("""
            result += f"""f=({self.stringfy_items(scores["fluency"])}), """
            result += f"""t=({self.stringfy_items(scores["truthfulness"])}), """
            result += f"""h={scores["helpfulness"]:.3f}"""
            if "helpfulness_results" in scores:
                keywords = []
                for r in scores["helpfulness_results"]:
                    if r[1] == 0.0:
                        keywords.append(r[0])
                    else:
                        keywords.append(f"""{r[0]}*{r[1]:.1f}""")
                if len(keywords) > 0:
                    result += f""" [{", ".join(keywords)}]"""
            result += f""")){extra}"""
        return result

    def process_result(self, result_path, output_path, *, force=False):
        if not force:
            print(f"Checking a result from {result_path}...", file=sys.stderr)
            # Calculate SHA-1 hash.
            with open(result_path, "rb") as f:
                result_hash = hashlib.sha1(f.read()).hexdigest()
            if os.path.exists(output_path):
                # If output file contains "<!-- result.json: SHA1 -->", skip it.
                with open(output_path) as f:
                    if f"<!-- result.json: {result_hash} -->" in f.read():
                        print(f"Skipping a result from {result_path}...", file=sys.stderr)
                        return

        print(f"Loading a result from {result_path}...", file=sys.stderr)
        with open(result_path) as f:
            result = json.load(f)

        print(f"Writing a report to {output_path}...", file=sys.stderr)
        with open(output_path + ".tmp", "w") as f:
            f.write(f"""# Model: {result["config"]["model"]}\n\n""")
            f.write(f"""<!-- result.json: {result_hash} -->\n\n""")

            f.write(
                self.stringfy_scores(
                    result["scores"],
                    long=True,
                    extra=f""" (±{result["score_std"]:.4f}/√{result["num_trials"]})""",
                )
            )

            f.write(f"""## Questions\n\n""")

            f.write("| Question | Score | Length |\n")
            f.write("|----------|-------|--------|\n")
            for d in self.metadata.values():
                question_id = d["question_id"]
                question = d["question"]
                r = result["questions"][question_id]
                f.write(
                    f"""| [{question_id}. {question}](#{question_id}) """
                    f"""| <code>{r["score"]:.4f} (±{r["score_std"]:.4f})</code> """
                    f"""| <code>{r["length"]:.1f} (±{r["length_std"]:.1f})</code> |\n"""
                )
            f.write("\n")

            config_data = json.dumps(result["config"], ensure_ascii=False)
            f.write(f"""## Config\n\n```json\n{config_data}\n```\n\n""")

            for md in self.metadata.values():
                question_id = md["question_id"]
                question = md["question"]
                f.write(f"""## <a name="{question_id}"></a>{question_id}. {question}\n\n""")
                r = result["questions"][question_id]
                f.write(
                    self.stringfy_scores(r["scores"], long=True, extra=f""" (±{r["score_std"]})""")
                )
                f.write(f"""- Length: {r["length"]} (±{r["length_std"]})\n\n""")
                f.write(f"""<dl>\n""")
                for a in r["samples"]:
                    f.write(f"""<dt>{self.stringfy_scores(a["scores"], long=False)}</dt>\n""")
                    answer = f"""^{a["answer"]}$"""
                    dist = [0 for _ in range(len(answer))]
                    total = sum(len(xs) for xs in md["answers"].values())
                    for i in range(len(dist) - 3 + 1):
                        t = answer[i : i + 3]
                        dist[i] = sum(
                            sum(1 if t in x else 0 for x in xs) for xs in md["answers"].values()
                        )
                    ngram_dist = []
                    for i in range(len(dist)):
                        ngram_dist.append(max(dist[max(0, i - 3 + 1) : i + 1]))
                    s = ""
                    for c, d in zip(answer[1:-1], ngram_dist[1:-1]):
                        # If c is a special character (e.g., "\n"), encode it.
                        if ord(c) < 32:
                            c = repr(c).strip("'\"")
                        c = html.escape(c)
                        # If c is a Markdown special character, encode it.
                        c = c.replace("\\", "&#92;")
                        if c in "_*`{}[]()#+-.!|":
                            c = f"&#{ord(c)};"
                        if d > total * 0.1:
                            s += f"<b>{c}</b>"
                        elif d < total * 0.005:
                            s += f"<s>{c}</s>"
                        else:
                            s += c
                    s = s.replace("</b><b>", "").replace("</s><s>", "")
                    f.write(f"""<dd>{s}</dd>\n""")
                f.write(f"""</dl>\n\n""")
        os.rename(output_path + ".tmp", output_path)
        print(f"Finished writing a report to {output_path}.", file=sys.stderr)

    def leaderboard(self):
        print("Loading results...", file=sys.stderr)
        results = {}
        for result_path in self.result_paths:
            with open(result_path) as f:
                result = json.load(f)
                results[result_path] = result

        print("Writing a leaderboard...", file=sys.stderr)
        buffer = ""
        table = []
        ranking = list(results.keys())
        ranking.sort(key=lambda x: results[x]["score"], reverse=True)
        rank = 1
        for i, name in enumerate(ranking):
            result = results[name]
            icon = "❔"
            mode = result["config"].get("mode", "unknown")
            model = result["config"].get("model", "unknown")
            if model == "system/ground-truth":
                icon = "👑"
            elif model == "system/criteria":
                icon = "🎯"
            if mode in ("qa", "chat"):
                icon = "💬"
            elif mode == "completion":
                icon = "🟢"
            if len(model) > 40:
                model = model[:37] + "..."
            table.append(
                {
                    "Rank": f"{rank}" if mode != "system" else "N/A",
                    "Score": f"{result['score']:.4f} (±{result['score_std']:.4f}/√{result['num_trials']})",
                    "Model": f"""{icon} {model}""",
                    "Length": f"{result['length']:.1f} (±{result['length_std']:.1f})",
                    "Fluency": f"""{sum(result["scores"]["fluency"].values()):.3f}""",
                    "Truthfulness": f"""{sum(result["scores"]["truthfulness"].values()):.3f}""",
                    "Helpfulness": f"""{result["scores"]["helpfulness"]:.3f}""",
                }
            )
            if mode != "system":
                rank += 1
        for key in table[0]:
            width = max(sum(1 if ord(x) < 128 else 2 for x in row[key]) for row in table)
            label = key + "&nbsp;" * max(0, width - len(key))
            buffer += f"| <code>{label}</code> "
        buffer += "|\n"
        for key in table[0]:
            buffer += "|-----"
        buffer += "|\n"
        for i, row in enumerate(table):
            report_path = os.path.join(os.path.dirname(ranking[i]), "README.md")
            for k, v in row.items():
                if k == "Model":
                    buffer += f"| <code>[{v}]({report_path})</code> "
                else:
                    buffer += f"| <code>{v}</code> "
            buffer += "|\n"

        with open("README.md") as f:
            template = f.read()
        with open("README.md.tmp", "w") as f:
            f.write(
                re.sub(
                    r"<!-- leaderboard -->(.*)<!-- /leaderboard -->",
                    r"<!-- leaderboard --><!-- /leaderboard -->",
                    template,
                    flags=re.DOTALL,
                ).replace(
                    "<!-- leaderboard --><!-- /leaderboard -->",
                    f"<!-- leaderboard -->\n{buffer}\n<!-- /leaderboard -->",
                )
            )
        os.rename("README.md.tmp", "README.md")
        print("Finished writing a leaderboard.", file=sys.stderr)

    def run(self, *, force=False):
        with futures.ProcessPoolExecutor(max_workers=32) as executor:
            fs = []
            for result_path in self.result_paths:
                output_path = os.path.join(os.path.dirname(result_path), "README.md")
                fs.append(
                    executor.submit(self.process_result, result_path, output_path, force=force)
                )
            fs.append(executor.submit(self.leaderboard))
            for f in futures.as_completed(fs):
                f.result()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--force", action="store_true")
    args = argparser.parse_args()
    PgenReporter().run(force=args.force)
