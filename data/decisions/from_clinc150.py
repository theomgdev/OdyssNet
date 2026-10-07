"""CLINC150 -> the decision JSONL format experiment_system_one.py reads.

A converter, not part of the training path: the script knows nothing about any
dataset, so preparing one is a separate job that lives next to its data.

    python from_clinc150.py                 # writes train/val/test.jsonl here

Source: https://github.com/clinc/oos-eval (Larson et al., EMNLP 2019).
150 intents over 10 domains, plus out-of-scope queries written to look like
the in-scope ones. Those become questions whose answer is `none of these`,
which is how a rejection reaches the model: as data, not as a head.
"""

import json
import pathlib
import random
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "clinc150"
URLS = {
    "data_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    "domains.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json",
}

#: Several phrasings per question kind, so a model that memorises one wording
#: scores no better than one that reads the question. Which half a phrasing
#: lands in is decided by the split, so the test set asks in words training
#: never used.
PHRASINGS = {
    "intent": ["what does the user want to do?",
               "which action is the user asking for?",
               "identify the user's intent",
               "which task should be run for this message?"],
    "domain": ["which area does this request belong to?",
               "what kind of service is being asked for?",
               "classify the topic of this message"],
    "scope": ["can this assistant handle that request?",
              "is this request in scope?",
              "does the assistant have a skill for this?"],
}

OPTIONS_PER_INTENT = 8          # wrong options sampled alongside the right one
REJECT = "none of these"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = RAW / name
        if not path.exists():
            print(f"fetching {name} ...")
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_bytes(r.read())
    return (json.loads((RAW / "data_full.json").read_text(encoding="utf-8")),
            json.loads((RAW / "domains.json").read_text(encoding="utf-8")))


def human(name):
    return name.replace("_", " ")


def convert(raw, domains, split, oos_split, phrasing_half, seed):
    rng = random.Random(seed)
    intent_domain = {i: d for d, members in domains.items() for i in members}
    intents = sorted(intent_domain)
    domain_names = sorted(domains)

    def phrasing(kind):
        pool = PHRASINGS[kind]
        cut = max(1, len(pool) * 2 // 3)
        half = pool[:cut] if phrasing_half == "train" else pool[cut:] or pool
        return half[rng.randrange(len(half))]

    rows = []
    for text, intent in raw[split]:
        wrong = rng.sample([i for i in intents if i != intent],
                           OPTIONS_PER_INTENT - 1)
        options = [human(o) for o in wrong + [intent]]
        rng.shuffle(options)
        rows.append({"context": text, "questions": [
            {"q": phrasing("intent"),
             "options": options + [REJECT],
             "correct": options.index(human(intent))},
            {"q": phrasing("domain"),
             "options": [human(d) for d in domain_names],
             "correct": domain_names.index(intent_domain[intent])},
            {"q": phrasing("scope"), "options": ["yes", "no"], "correct": 0},
        ]})

    for text, _ in raw.get(oos_split, []):
        options = [human(o) for o in rng.sample(intents, OPTIONS_PER_INTENT)]
        rows.append({"context": text, "questions": [
            # Nothing fits, so the rejection option is the right answer.
            {"q": phrasing("intent"),
             "options": options + [REJECT],
             "correct": len(options)},
            {"q": phrasing("scope"), "options": ["yes", "no"], "correct": 1},
        ]})

    rng.shuffle(rows)
    return rows


def main():
    raw, domains = fetch()
    plan = (("train", "train", "oos_train", "train", 42),
            ("val", "val", "oos_val", "train", 43),
            # The test set asks in phrasings training never used, so a model
            # that pattern-matched a fixed wording shows up here.
            ("test", "test", "oos_test", "held_out", 44))
    for name, split, oos, half, seed in plan:
        rows = convert(raw, domains, split, oos, half, seed)
        path = HERE / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        questions = sum(len(r["questions"]) for r in rows)
        print(f"{path.name}: {len(rows):,} contexts, {questions:,} questions")


if __name__ == "__main__":
    main()
