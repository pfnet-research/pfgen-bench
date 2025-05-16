import datetime
import hashlib
import json
import os
import sys
import time
import typing

QUESTION_DATA: str = r"""
{"question": "時計の長針と短針が1日に重なる回数は？", "answer": "時計の長針と短針が1日に重なる回数は、22回です。これは、長針が1日に24周するのに対して、短針は1日に2周するためです。長針が短針を追い越す瞬間に重なるので、重なる回数は24-2=22回となります。"}
{"question": "つるかめ算について教えて。", "answer": "つるかめ算は、算数の文章題の解き方の一つで、つるとかめの合計数や足の本数などから、それぞれの数を求める方法です。連立方程式を使わずに、面積図などを用いて解くことができるため、中学受験でも出題されます。"}
{"question": "直角二等辺三角形の特徴を説明してください。", "answer": "直角二等辺三角形は、直角を挟む2つの辺が等しい三角形です。直角以外の角が45度になるのが特徴で、正方形を対角線で2等分した形でもあります。ピタゴラスの定理により、辺の長さの比は1:1:√2となります。"}
{"question": "算数と数学の違いは何ですか？", "answer": "算数は、小学校で学ぶ基礎的な計算や図形などを扱う科目で、日常生活に役立つ実用的な知識を身につけます。一方、数学は、中学校以降で学ぶより高度な計算や理論を扱う科目で、抽象的な概念や論理的思考を養います。"}
{"question": "ナメクジに塩をかけるとなぜ溶けてしまうの？", "answer": "ナメクジに塩をかけると溶けるように見えるのは、塩がナメクジの水分を奪うからです。ナメクジの体は約90%が水分で構成されており、塩をかけると浸透圧の作用により水分が外に出てしまい、体が縮んでしまいます。"}
{"question": "ミドリムシの特徴を教えて。", "answer": "ミドリムシは、体長約0.05mmの単細胞生物で、植物と動物の特徴を合わせ持っています。葉緑体を持ち光合成を行う一方で、鞭毛を使って移動することができます。健康食品やバイオ燃料としても注目されています。"}
{"question": "顕性と潜性の違いは？", "answer": "顕性と潜性の違いは、遺伝子の表現型の現れ方にあります。顕性（優性）形質は、遺伝子を1つだけ持っている場合でも現れる形質で、潜性（劣性）形質は、同じ形質の遺伝子を2つ持っている場合にのみ現れる形質です。"}
{"question": "スズムシの鳴き声について教えて。", "answer": "スズムシは、日本に生息するコオロギ科の昆虫で、夏から秋にかけて「リーン、リーン」と鳴きます。鳴き声はオスがメスを誘う求愛行動で、羽をこすり合わせて音を出します。日本の秋の風物詩として親しまれています。"}
{"question": "タマネギを切ると涙が出るのはなぜ？", "answer": "タマネギを切ると涙が出るのは、タマネギの細胞が破壊されるとアリナーゼという酵素が働いて、催涙物質であるsyn-プロパンチアール-S-オキシドが生成されるためです。この物質が目や鼻を刺激して涙が出ます。"}
{"question": "接触法について教えて。", "answer": "接触法とは、硫黄を燃焼させて二酸化硫黄を作り、それを酸化バナジウム(V)の触媒を用いて酸化させて三酸化硫黄を作り、これを硫酸に吸収させて発煙硫酸とし、最後に希硫酸で希釈して濃硫酸を得る工業的製法です。"}
{"question": "温泉卵と半熟卵の違いは何から生まれるの？", "answer": "温泉卵と半熟卵の違いは、加熱時間と温度によって生まれます。温泉卵は、65～70度程度で長時間加熱し、黄身と白身が半熟状態になります。半熟卵は、沸騰したお湯で短時間加熱し、黄身は半熟で白身は固まります。"}
{"question": "リトマス紙の使い方を教えて。", "answer": "リトマス紙は、液体の酸性・アルカリ性を調べるために使用される試験紙です。青色のリトマス紙は、酸性の液体に触れると赤色に変わります。一方、赤色のリトマス紙は、アルカリ性の液体に触れると青色に変わります。"}
{"question": "ドップラー効果について教えて。", "answer": "ドップラー効果は、音や光などの波が発生源や観測者の相対的な運動によって周波数が変化する現象です。例えば、救急車が近づくとサイレンの音が高く聞こえ、遠ざかると低く聞こえる現象がドップラー効果の一例です。"}
{"question": "超伝導とは何ですか？", "answer": "超伝導とは、ある種の物質が特定の温度以下で電気抵抗がゼロになる現象です。超伝導状態では、電流が損失なしに流れるため、強力な磁場を発生させることができ、MRIやリニアモーターカーなどに応用されています。"}
{"question": "虹はどうして虹色なの？", "answer": "虹は、太陽光が大気中の水滴で反射・屈折する際に発生する現象で、光が波長によって屈折率が異なるため、色が分かれて見えます。可視光線の中で波長が長いものから、赤、橙、黄、緑、青、藍、紫の順に色が見えます。"}
{"question": "カミオカンデは何を行う施設ですか？", "answer": "カミオカンデは、岐阜県飛騨市の神岡鉱山の地下深くに設置された観測装置です。1987年に大マゼラン星雲で発生した超新星爆発のニュートリノを検出し、小柴昌俊氏が2002年にノーベル物理学賞を受賞しました。"}
{"question": "日本はどうして地震が多いの？", "answer": "日本は、環太平洋火山帯に位置し、地殻変動が活発な地域です。また、太平洋プレート、フィリピン海プレート、北米プレート、ユーラシアプレートの4つのプレートの境界にあるため、地震が発生しやすくなっています。"}
{"question": "糸魚川静岡構造線とは何ですか？", "answer": "糸魚川静岡構造線とは、新潟県糸魚川市から静岡県静岡市まで続く大断層帯で、日本列島を東西に分ける地質境界です。フォッサマグナの西縁にあたり、北アメリカプレートとユーラシアプレートの境界にもなっています。"}
{"question": "夏はどうして暑いの？", "answer": "夏が暑い理由は、地球の自転軸が公転面に対して約23.4度傾いているからです。この傾きにより、夏は太陽の高度が高く、日照時間も長くなります。その結果、太陽から受ける熱エネルギーが増え、気温が上昇します。"}
{"question": "地球の歴史について教えて。", "answer": "地球は約46億年前に誕生し、その歴史は大きく分けて先カンブリア時代、古生代、中生代、新生代の4つの地質時代に区分されます。生命の誕生や進化、大陸の移動、気候変動などが起き、現在の地球の姿になりました。"}
{"question": "Rubyについて教えて。", "answer": "Rubyは、日本のまつもとゆきひろ氏が開発したオブジェクト指向プログラミング言語で、1995年に公開されました。シンプルな文法と高い生産性が特徴で、Webアプリケーション開発などで広く使われています。"}
{"question": "自然言語処理の主要な技術について教えて。", "answer": "自然言語処理の主要な技術には、形態素解析、構文解析、意味解析、文脈解析などがあります。これらの技術を組み合わせることで、機械が自然言語を理解し、自動翻訳や感情分析、質問応答などの応用が可能になります。"}
{"question": "オゾン層って何ですか？", "answer": "オゾン層は、地球の成層圏（高度約10～50km）に存在するオゾン濃度の高い層です。太陽からの有害な紫外線を吸収し、生物を保護する役割を果たしていますが、フロンガスなどの影響で破壊されることがあります。"}
{"question": "再生可能エネルギーとは何ですか？", "answer": "再生可能エネルギーとは、太陽光、風力、水力、地熱、バイオマスなど、自然の力を利用して得られるエネルギーです。枯渇する心配がなく、環境負荷も少ないことから、持続可能なエネルギー源として注目されています。"}
{"question": "四大公害病について教えて。", "answer": "四大公害病とは、日本の高度経済成長期に発生した公害病で、水俣病、新潟水俣病、四日市ぜんそく、イタイイタイ病の4つを指します。いずれも産業活動に伴う有害物質の排出が原因で、多くの人々が被害を受けました。"}
{"question": "夢の島の歴史について教えて。", "answer": "夢の島は、東京都江東区にある人工島で、1957年からゴミの最終処分場として利用されていました。1967年に埋め立てが終了し、現在では熱帯植物館やスポーツ施設などが整備され、公園として親しまれています。"}
{"question": "競技かるたとは何ですか？", "answer": "競技かるたとは、小倉百人一首を用いた日本の伝統的なかるた遊びを競技化したものです。読み手が上の句を読み、競技者は下の句が書かれた札を取る速さや正確さを競います。全国大会も開催され、人気を集めています。"}
{"question": "漢文における返り点について教えて。", "answer": "返り点とは、漢文を読みやすくするために付けられる記号のことです。返り点には、レ点、一二三点、上中下点、甲乙丙点、天地人点などがあり、これらを使うことで、漢文を日本語の語順に合わせて読むことができます。"}
{"question": "擬音語と擬態語の違いは何ですか？", "answer": "擬音語は、実際の音を言葉で表現したものです。例えば、「ワンワン」「ザーザー」などがあります。一方、擬態語は、実際の音ではなく、状態や感情を表す言葉です。例えば、「キラキラ」「ドキドキ」などがあります。"}
{"question": "重箱読みとは何ですか？", "answer": "重箱読みとは、熟語の読み方の一つで、「重箱（ジュウばこ）」のように、前の漢字を音読み、後の漢字を訓読みする読み方のことです。逆に、前の漢字を訓読み、後の漢字を音読みする読み方を「湯桶読み」といいます。"}
{"question": "日本の開国について教えて。", "answer": "日本の開国は、1853年にペリー提督率いるアメリカの黒船が浦賀に来航したことをきっかけに始まりました。その後、日米和親条約や日米修好通商条約などを締結し、鎖国政策を改め、諸外国との交流を開始しました。"}
{"question": "関ヶ原の戦いについて教えて。", "answer": "関ヶ原の戦いとは、1600年に現在の岐阜県関ヶ原町で行われた合戦で、徳川家康率いる東軍と石田三成率いる西軍が激突した天下分け目の戦いです。徳川家康が率いる東軍が勝利し、徳川幕府の樹立につながりました。"}
{"question": "日本の東西南北端点について教えて。", "answer": "日本の東西南北端点は、北端が択捉島（北海道）、南端が沖ノ鳥島（東京都）、東端が南鳥島（東京都）、西端が与那国島（沖縄県）です。ただし、択捉島はロシアとの領土問題があり、日本の実効支配が及んでいません。"}
{"question": "瀬戸内海式気候とは何ですか？", "answer": "瀬戸内海式気候とは、日本の気候区分の一つで、瀬戸内海沿岸地域に見られる気候です。年間を通して降水量が少なく、日照時間が長いのが特徴です。また、季節風が山地に遮られ、年間を通して穏やかな気候となります。"}
{"question": "天皇はどのような役割をもっていますか？", "answer": "天皇は日本の君主であり、日本国憲法第1条で「日本国の象徴であり日本国民統合の象徴」と定められています。政治的な権限は持っていませんが、国事行為や公的行事への出席など、国家の重要な役割を果たしています。"}
{"question": "三権分立とは何ですか？", "answer": "三権分立とは、国家権力を立法府（国会）、行政府（内閣）、司法府（裁判所）の3つの機関に分け、互いに抑制と均衡を図る仕組みです。これにより、権力の集中や濫用を防ぎ、国民の権利と自由を守ることができます。"}
{"question": "日本銀行の役割は何ですか？", "answer": "日本銀行の主な役割は、日本の中央銀行として物価の安定と金融システムの安定を図り、日本経済の健全な発展に貢献することです。具体的には、紙幣の発行、金融政策の実施、金融機関への資金供給などを行っています。"}
{"question": "信用取引と先物取引の違いは何ですか？", "answer": "信用取引は、証券会社から資金や株式を借りて株式を売買する取引で、借りた資金や株式を期限内に返済する必要があります。一方、先物取引は、将来の特定の日に特定の商品を特定の価格で売買する契約を結ぶ取引です。"}
{"question": "日本脳炎とはどのような感染症ですか？", "answer": "日本脳炎は、日本脳炎ウイルスによって引き起こされる感染症で、蚊を介して人に感染します。高熱、頭痛、嘔吐、意識障害などの症状が現れ、重症化すると死に至ることもあります。ワクチン接種による予防が可能です。"}
{"question": "柔道と合気道の違いを教えて。", "answer": "柔道と合気道の違いは、技術や目的にあります。柔道は、投げ技や固め技を使って相手を倒すことを目的とする競技です。一方、合気道は、相手の力を利用して技をかけることで、相手を制することを目的とする武道です。"}
{"question": "葛根湯とは何ですか？", "answer": "葛根湯は、漢方薬の一つで、風邪の初期症状や肩こり、頭痛などの症状に用いられます。葛根、麻黄、桂枝、生姜、甘草、芍薬、大棗の7つの生薬からなり、体を温めて発汗を促し、風邪の症状を改善する効果があります。"}
{"question": "必須アミノ酸とは何ですか？", "answer": "必須アミノ酸は、体内で合成できず、食事から摂取する必要がある9つのアミノ酸です。バリン、ロイシン、イソロイシン、リジン、メチオニン、フェニルアラニン、トレオニン、トリプトファン、ヒスチジンがあります。"}
{"question": "天空の城ラピュタはどのような作品ですか？", "answer": "天空の城ラピュタは、宮崎駿監督によるスタジオジブリ制作の長編アニメーション映画で、1986年に公開されました。空に浮かぶ伝説の島「ラピュタ」を舞台に、少年パズーと少女シータの冒険と成長を描いています。"}
{"question": "走れメロスはどのような作品ですか？", "answer": "走れメロスは、太宰治が1940年に発表した短編小説です。王に人質として差し出された親友を救うため、困難な状況でも走り続けるメロスの姿を描いています。友情や信頼の大切さを伝える作品として知られています。"}
{"question": "山田耕筰は何をした人ですか？", "answer": "山田耕筰は、日本の作曲家、指揮者であり、日本における西洋音楽の普及に貢献しました。「からたちの花」「赤とんぼ」などの童謡を作曲し、日本初の交響楽団を設立するなど、日本の音楽界に大きな影響を与えました。"}
{"question": "宝塚歌劇団の特徴は？", "answer": "宝塚歌劇団は、兵庫県宝塚市に本拠地を置く、未婚の女性のみで構成された歌劇団です。男役と娘役による華やかなショーやミュージカルが特徴で、1914年に初公演を行って以来、多くのファンを魅了し続けています。"}
{"question": "春分の日と秋分の日はどのように決まるの？", "answer": "春分の日と秋分の日は、太陽が黄道上の春分点と秋分点を通過する日と定められており、毎年2月に国立天文台が発表します。春分の日は3月20日頃、秋分の日は9月23日頃で、昼と夜の長さがほぼ同じになる日です。"}
{"question": "七草がゆについて教えて。", "answer": "七草がゆは、1月7日の「人日の節句」に食べる日本の伝統的な行事食です。春の七草（セリ、ナズナ、ゴギョウ、ハコベラ、ホトケノザ、スズナ、スズシロ）を入れたおかゆを食べることで、1年の無病息災を願います。"}
{"question": "神社と寺の違いについて教えて。", "answer": "神社は日本古来の宗教である神道の信仰施設で、神々を祀る場所です。一方、寺は仏教の信仰施設で、仏像や経典が祀られています。神社は鳥居があり、神職が神事を行い、寺は仏像があり、僧侶が仏教の教えを説きます。"}
{"question": "神在月とは何ですか？", "answer": "神在月とは、旧暦10月のことを指し、全国の八百万の神々が出雲大社に集まり、縁結びの神議り（かむはかり）が行われるとされる月です。出雲地方では「神在月」と呼びますが、他の地域では「神無月」と呼ばれます。"}
"""  # noqa: E501
QUESTIONS: list[dict[str, str]] = []


def get_questions() -> list[dict[str, str]]:
    global QUESTIONS
    if len(QUESTIONS) == 0:
        for line in QUESTION_DATA.strip().split("\n"):
            QUESTIONS.append(json.loads(line))
    return QUESTIONS


def generate_examples(
    question: dict[str, str], trial: int, num_examples: int = 20, seed: str = ""
) -> str:
    questions = get_questions()
    examples = [q for q in questions if question["question"] != q["question"]]
    examples.sort(
        key=lambda x: hashlib.sha1(f"{seed}::{trial}::{x['question']}".encode()).hexdigest()
    )
    prompt = ""
    for example in examples[:num_examples]:
        prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
    return prompt


def generate_task(
    question: dict[str, str],
    mode: str,
    trial: int,
    num_examples: int = 20,
    prefix: str = "",
    seed: str = "",
) -> dict[str, str]:
    if mode == "chat":
        system_prompt = "例と同様の文体及び文字数で、ユーザの質問に1行で答えてください。\n\n"
    elif mode == "qa":
        system_prompt = "例と同様の文体及び文字数で、質問に1行で答えてください。\n\n"
    else:
        system_prompt = ""
    system_prompt += "## 回答例\n"
    system_prompt += generate_examples(
        question, trial, num_examples=num_examples, seed=seed
    ).strip()
    user_prompt = f"""Q: {question["question"]}\n"""
    if mode == "completion":
        user_prompt += "A:"
    task: dict[str, typing.Any] = {"question": question["question"]}
    if mode == "chat":
        task["system_prompt"] = prefix + system_prompt.strip()
        task["user_prompt"] = user_prompt.strip()
    elif mode == "qa":
        task["prompt"] = prefix + system_prompt.strip() + "\n\n## 質問\n" + user_prompt.strip()
    elif mode == "completion":
        task["prompt"] = prefix + system_prompt.strip() + "\n\n" + user_prompt.strip()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    task["seed"] = int(hashlib.sha1(f"{seed}::{trial}".encode()).hexdigest(), 16) % (2**31)
    return task


def run_tasks(
    mode: str,
    callback: typing.Callable[
        [list[dict[str, str]], dict[str, typing.Any]],
        typing.Iterable[str | None],
    ],
    *,
    engine: str,
    model: str,
    num_examples: int = 20,
    num_trials: int = 100,
    seed: str = "",
    **parameters: typing.Any,
) -> None:
    questions = get_questions()
    parameters["engine"] = engine
    parameters["model"] = model
    parameters["mode"] = mode
    parameters["num_examples"] = num_examples
    parameters["stop"] = ["Q:"]
    if mode == "completion":
        parameters["stop"].append("\n\n")
    parameters["max_tokens"] = 300
    config_parameters = [(k, v) for k, v in parameters.items() if not k.startswith("_")]
    config = json.dumps(dict(sorted(config_parameters)), ensure_ascii=False)
    config_hash = hashlib.sha1(config.encode()).hexdigest()[:7]
    result_dir = os.path.join(os.path.dirname(__file__), "result", model, config_hash)
    print(f"Result directory: {result_dir}", file=sys.stderr)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "config.json"), "w", encoding="utf-8") as f:
        print(config, file=f)
    trial_path = ""
    buf = ""
    for trial in range(1, num_trials + 1):
        trial_path = os.path.join(result_dir, f"trial_{trial:03d}.jsonl")
        if os.path.exists(trial_path):
            continue
        print(f"Starting a trial: {trial}", file=sys.stderr)
        if buf == "":
            outputs: dict[str, str] = {}
            for _ in range(10):
                tasks: list[dict[str, str]] = []
                task_questions: list[str] = []
                for q_info in questions:
                    if q_info["question"] in outputs:
                        continue
                    task_questions.append(q_info["question"])
                    tasks.append(
                        generate_task(
                            q_info,
                            mode,
                            trial=trial,
                            num_examples=num_examples,
                            prefix=parameters.get("prefix", "")
                            .encode("utf-8")
                            .decode("unicode_escape")
                            .encode("latin1")
                            .decode("utf-8"),
                            seed=seed,
                        )
                    )
                if len(tasks) == 0:
                    break
                print(f"parameters: {parameters}")
                for q, a in zip(task_questions, callback(tasks, parameters)):
                    if a is None or a == "":
                        print(f"Failed to get an answer for: {q}", file=sys.stderr)
                        time.sleep(3)
                        continue
                    if mode in ("chat", "qa") and "A:" in a:
                        a = a.split("A:", 1)[1].strip()
                    result = {
                        "question": q,
                        "answer": a.strip(),
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    output = json.dumps(result, ensure_ascii=False)
                    print(f"Result: {output}", file=sys.stderr)
                    outputs[q] = output
            for q_info in questions:
                if q_info["question"] not in outputs:
                    raise RuntimeError(f"""Missing result for: {q_info["question"]}""")
                buf += outputs[q_info["question"]] + "\n"
        if os.path.exists(trial_path):
            print(f"File already exists: {trial_path}", file=sys.stderr)
            continue
        tmp_file = f"{trial_path}.{datetime.datetime.now().microsecond:06d}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(buf)
        os.rename(tmp_file, trial_path)
        print(f"Saved: {trial_path}", file=sys.stderr)
        buf = ""
