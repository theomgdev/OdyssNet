"""A graded synthetic corpus in the decision JSONL format, no download.

A converter, not part of the training path: the script knows nothing about any
dataset, so preparing one is a separate job that lives next to its data.

    python synthetic_basics.py                  # writes train/val/test.jsonl
    python synthetic_basics.py --contexts 40000 --out-dir ../mixed

The point is a floor, not a benchmark. CLINC150 asks one hard thing — pick an
intent out of 150 — and a model that cannot read at all scores much like one
that half can, so a flat loss curve says nothing about which. These questions
are easy enough that failing them means the bytes are not being read, and they
come in bands by how much work the answer takes, not by which scene produced
it:

    1  stated       the answer is a word or a claim sitting in the context
    2  unanswerable the context does not settle it, so no option is right
    3  recalled     the context plus one thing a reader is expected to know
    4  compared     two quantities in the context, weighed against each other
    5  inferred     one hop: a rule in the context applied to a fact in it

Every question carries its band, so accuracy can be read per band rather than
as one average that hides which half carried it.

Four properties the generator keeps, because each one closes a way of scoring
well without reading:

* A scene is built first and several questions are asked about it, so the
  stored-state amortisation the protocol exists for is actually exercised.
* An option set is never a fixed vocabulary. Wrong options are sampled per
  question, the correct position is uniform, and K varies from 2 to 10.
* Some questions are answered by none of the options and say so with
  `correct: -1` — the branch the rest of the corpus never reaches, and the
  only one where a model has to decline rather than pick the least wrong.
* Splits get disjoint phrasings and disjoint entity pools, so a model that
  memorised a wording or a noun scores at chance on test.
"""

import argparse
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Vocabulary. Split so test uses words training never saw.                    #
# --------------------------------------------------------------------------- #

FRUIT = ["apple", "banana", "cherry", "grape", "lemon", "mango", "peach",
         "plum", "melon", "orange", "apricot", "fig"]
VEGETABLE = ["carrot", "potato", "onion", "pepper", "cabbage", "spinach",
             "turnip", "leek", "radish", "pumpkin", "celery", "beet"]
ANIMAL = ["cat", "dog", "horse", "rabbit", "sheep", "goat", "duck", "goose",
          "mouse", "donkey", "camel", "pigeon"]
TOOL = ["hammer", "wrench", "ladder", "shovel", "broom", "rake", "drill",
        "saw", "brush", "chisel", "pliers", "clamp"]
NAME = ["john", "mary", "ahmet", "sara", "omar", "lena", "pavel", "nadia",
        "tariq", "elif", "diego", "yuki", "rosa", "kemal", "ines", "bilal"]
PLACE = ["the market", "the garden", "the kitchen", "the office",
         "the station", "the library", "the harbour", "the workshop",
         "the cellar", "the rooftop", "the courtyard", "the bakery",
         "the shipyard", "the orchard", "the attic", "the pier"]
COLOUR = ["red", "green", "blue", "yellow", "brown", "grey", "purple",
          "orange", "black", "white", "pink", "gold"]

#: Noise sentences, so the answer is never the only thing present.
FILLER = [
    "the weather was unremarkable that morning",
    "nobody had remembered to close the window",
    "a radio played somewhere down the street",
    "the lights flickered once and settled",
    "it had rained during the night",
    "the floor needed sweeping again",
    "a letter arrived with no return address",
    "the kettle was still warm",
    "two chairs stood against the far wall",
    "the clock on the shelf ran slow",
]

CATEGORY = {"fruit": FRUIT, "vegetable": VEGETABLE, "animal": ANIMAL,
            "tool": TOOL}
#: Category words the scenes never use, so a question can be about something
#: the context genuinely does not settle.
EXTRA_CATEGORIES = ["drink", "building", "weed", "mineral"]
REJECT = ["none of these", "no option is correct", "none of the above"]

PHRASING = {
    "carried":  {"train": ["what did {who} carry?", "which item did {who} bring?",
                           "name the thing {who} had"],
                 "held_out": ["what was {who} holding?"]},
    "where":    {"train": ["where was {who}?", "which place was {who} in?"],
                 "held_out": ["name the place {who} went to"]},
    "is_a":     {"train": ["is the {thing} a {cat}?",
                           "would you call the {thing} a {cat}?"],
                 "held_out": ["does the {thing} count as a {cat}?"]},
    "category": {"train": ["what is a {thing}?",
                           "which group does the {thing} belong to?"],
                 "held_out": ["classify the {thing}"]},
    "count":    {"train": ["how many {what} does {who} have?",
                           "count {who}'s {what}"],
                 "held_out": ["give the number of {what} {who} has"]},
    "sum":      {"train": ["what is {a} + {b}?", "add {a} and {b}"],
                 "held_out": ["{a} plus {b} is what?"]},
    "more":     {"train": ["who had more?", "which of them had the larger number?"],
                 "held_out": ["name the one with more"]},
    "colour":   {"train": ["so is the {thing} {adj}?",
                           "does that make the {thing} {adj}?"],
                 "held_out": ["given the rule, is the {thing} {adj}?"]},
    "unknown":  {"train": ["what colour is the {thing}?",
                           "which colour does the {thing} have?"],
                 "held_out": ["name the {thing}'s colour"]},
}


def halve(pool, half):
    cut = len(pool) * 3 // 4
    return pool[:cut] if half == "train" else pool[cut:]


# --------------------------------------------------------------------------- #
# Scenes: a few sentences plus the facts a question may be built from.        #
# --------------------------------------------------------------------------- #

class Generator:
    def __init__(self, rng, half):
        self.rng = rng
        self.half = half
        self.name = halve(NAME, half)
        self.place = halve(PLACE, half)
        self.colour = halve(COLOUR, half)
        self.cat = {k: halve(v, half) for k, v in CATEGORY.items()}
        self.things = [(t, c) for c, pool in self.cat.items() for t in pool]
        self.all_things = [t for t, _ in self.things]
        self.cat_names = list(CATEGORY) + EXTRA_CATEGORIES

    def say(self, kind, **kw):
        pool = PHRASING[kind][self.half] or PHRASING[kind]["train"]
        return pool[self.rng.randrange(len(pool))].format(**kw)

    def spread(self, right, pool, k):
        """k options, one of them `right`, position uniform."""
        wrong = [o for o in pool if o != right]
        self.rng.shuffle(wrong)
        options = wrong[: max(1, k - 1)] + [right]
        self.rng.shuffle(options)
        return options, options.index(right)

    def nowhere(self, pool, k, avoid=()):
        """k options none of which is right — the `correct: -1` case."""
        pool = [o for o in pool if o not in avoid]
        self.rng.shuffle(pool)
        return pool[:k], -1

    def numeric(self, right, k, hi=20):
        return self.spread(str(right), [str(v) for v in range(hi + 1)], k)

    def noise(self, n):
        return self.rng.sample(FILLER, n)

    # -- scene builders: (context, [question factories]) -------------------- #

    def scene_errand(self):
        """Someone takes something somewhere. Bands 1-2."""
        rng = self.rng
        who, other = rng.sample(self.name, 2)
        thing, cat = rng.choice(self.things)
        where = rng.choice(self.place)
        n = self.noise(rng.randint(1, 3))
        ctx = ". ".join(n[:1] + [f"{who} carried a {thing} to {where}"] + n[1:])

        def q_thing():
            o, r = self.spread(thing, self.all_things, rng.randint(3, 8))
            return self.say("carried", who=who), o, r, 1

        def q_place():
            o, r = self.spread(where, self.place,
                               rng.randint(3, len(self.place)))
            return self.say("where", who=who), o, r, 1

        def q_other():
            # `other` is not in the scene, so nothing can be said about them.
            o, r = self.nowhere(self.all_things, rng.randint(3, 6),
                                avoid=(thing,))
            return self.say("carried", who=other), o, r, 2

        def q_category():
            # The category is not in the context at all — the thing's name
            # has to be known, not located.
            o, r = self.spread(cat, self.cat_names, rng.randint(3, 6))
            return self.say("category", thing=thing), o, r, 3

        return ctx, [q_thing, q_place, q_other, q_category]

    def scene_denial(self):
        """The context denies a category. Bands 1-3."""
        rng = self.rng
        thing, real = rng.choice(self.things)
        wrong = rng.choice([c for c in self.cat_names if c != real])
        who = rng.choice(self.name)
        # Two filler sentences and a bystander, so the scene does not collapse
        # onto a handful of repeated strings — `group_by_context` merges rows
        # that share a context, and a duplicate scene would silently become
        # one row carrying nine questions.
        n = self.noise(2)
        ctx = ". ".join([n[0], f"{who} said a {thing} is not a {wrong}", n[1]])

        def q_denied():
            # Stated outright: the context says exactly this.
            return self.say("is_a", thing=thing, cat=wrong), ["yes", "no"], 1, 1

        def q_true():
            # The denial is about another category, so this one needs the
            # world rather than the sentence — a harder read.
            return self.say("is_a", thing=thing, cat=real), ["yes", "no"], 0, 3

        def q_unsettled():
            # The denial says what it is not; the list offers only other
            # wrong categories, so no option is right.
            o, r = self.nowhere([c for c in self.cat_names if c != real],
                                rng.randint(2, 5))
            return self.say("category", thing=thing), o, r, 2

        return ctx, [q_denied, q_true, q_unsettled]

    def scene_tally(self):
        """Two people hold countable things. Bands 3-4."""
        rng = self.rng
        who, other = rng.sample(self.name, 2)
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        while a == b:
            b = rng.randint(1, 9)
        cat = rng.choice(list(self.cat))
        ctx = (f"{who} has {a} {cat}s. {other} has {b} {cat}s. "
               + rng.choice(FILLER))

        def q_count_a():
            # The number is written in the context; finding it is the work.
            o, r = self.numeric(a, rng.randint(3, 7), 9)
            return self.say("count", who=who, what=f"{cat}s"), o, r, 1

        def q_sum():
            o, r = self.numeric(a + b, rng.randint(4, 9))
            return self.say("sum", a=a, b=b), o, r, 3

        def q_more():
            o, r = self.spread(who if a > b else other, self.name,
                               rng.randint(2, 5))
            return self.say("more"), o, r, 4

        return ctx, [q_count_a, q_sum, q_more]

    def scene_rule(self):
        """A rule plus a fact, or a rule that does not reach. Band 5."""
        rng = self.rng
        thing, cat = rng.choice(self.things)
        adj = rng.choice(self.colour)
        holds = rng.random() < 0.5
        who = rng.choice(self.name)
        n = self.noise(2)
        ctx = ". ".join([n[0], f"{who} knows every {cat} is {adj}",
                         f"the {thing} is " + ("a " if holds else "not a ")
                         + cat, n[1]])

        def q_follows():
            return (self.say("colour", thing=thing, adj=adj),
                    ["yes", "no"], 0 if holds else 1, 5)

        def q_colour():
            if holds:
                o, r = self.spread(adj, self.colour, rng.randint(3, 6))
                band = 5
            else:
                # The rule never reaches the thing, so its colour is unstated.
                o, r = self.nowhere(self.colour, rng.randint(2, 5))
                band = 2
            return self.say("unknown", thing=thing), o, r, band

        return ctx, [q_follows, q_colour]


#: How often each scene is drawn. Easy scenes stay frequent on purpose: the
#: corpus is a curriculum, and a model that loses retrieval has lost the thing
#: the harder bands are built on.
SCENES = (("scene_errand", 4), ("scene_denial", 2),
          ("scene_tally", 4), ("scene_rule", 3))


def build(n, half, seed):
    rng = random.Random(seed)
    gen = Generator(rng, half)
    pool = [name for name, w in SCENES for _ in range(w)]

    rows = []
    for _ in range(n):
        ctx, factories = getattr(gen, rng.choice(pool))()
        # Several questions off one context: that is what the stored state is
        # for, and a corpus of one question per context would never show it.
        how_many = min(len(factories), rng.randint(1, 3))
        questions = []
        for make in rng.sample(factories, how_many):
            q, options, correct, band = make()
            questions.append({"q": q, "options": options, "correct": correct,
                              "band": band})
        rows.append({"context": ctx, "questions": questions})
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contexts", type=int, default=24000, metavar="N",
                   help="training contexts; val and test get a tenth each "
                        "(default: %(default)s)")
    p.add_argument("--out-dir", default=str(HERE), metavar="DIR",
                   help="where the three JSONL files go (default: here)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tenth = max(1, a.contexts // 10)
    plan = (("train", a.contexts, "train", a.seed),
            ("val", tenth, "train", a.seed + 1),
            # Test asks in phrasings and about entities training never saw, so
            # a model that memorised either shows up here rather than later.
            ("test", tenth, "held_out", a.seed + 2))

    for name, n, half, seed in plan:
        rows = build(n, half, seed)
        path = out / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        qs = [q for r in rows for q in r["questions"]]
        per_band = {b: sum(1 for q in qs if q["band"] == b)
                    for b in sorted({q["band"] for q in qs})}
        k = [len(q["options"]) for q in qs]
        none = sum(1 for q in qs if q["correct"] < 0)
        print(f"{path.name}: {len(rows):,} contexts | {len(qs):,} questions "
              f"({len(qs)/len(rows):.1f} per context) | K {min(k)}-{max(k)} | "
              f"{none:,} unanswerable ({100*none/len(qs):.0f}%)")
        print(f"   bands {per_band}")


if __name__ == "__main__":
    main()
