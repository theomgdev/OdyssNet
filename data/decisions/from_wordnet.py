"""English word knowledge in the decision JSONL format, from WordNet.

A converter, not part of the training path: the harness knows nothing about any
dataset, so preparing one is a separate job that lives next to its data.

    python from_wordnet.py                          # writes train/val/test.jsonl
    python from_wordnet.py --contexts 200000 --sentences data/wikisent2.txt

The synthetic and CLINC150 corpora teach the protocol — find a stated fact,
decline when nothing fits, pick an intent. Neither teaches what a word means,
so a model trained on them reads bytes without the meanings behind them. This
one asks, off a sentence or a definition:

    cloze     a word is blanked out of a real sentence; which word was it
    usage     which of these sentences uses the word correctly
    gloss     which definition belongs to this word
    name      which word does this definition describe
    kind      what kind of thing is this (immediate hypernym)
    same      which of these means the same
    opposite  which of these means the opposite

Difficulty is the distractor pool, not the question. A wrong option drawn from
the whole dictionary is settled by topic; one drawn from the target's own
siblings under a shared hypernym needs the distinction itself. Bands name which
pool a question used, so accuracy reads per band:

    1  distant   distractors sampled across the dictionary
    2  related   distractors share the target's part of speech and frequency
    3  sibling   distractors share the target's immediate hypernym
    4  sense     distractors are other senses of the target's own word
    5  rejected  no option is right, and saying so is the answer

Five properties the generator keeps, because each one closes a way of scoring
well without knowing the word:

* Polysemy decides what is askable. "Which definition belongs to `bank`" has
  several right answers, so definition and usage questions take monosemous
  targets only; cloze takes any word, since the sentence fixes the sense.
* A gloss that spells its own lemma out is dropped rather than asked.
* Frequency-matched distractors: a rare correct answer among common wrong ones
  is answerable from frequency alone.
* Splits are disjoint by *answer word*, so a word that answers a training
  question never answers a test one — including as a hypernym or a synonym,
  which is where a synset-level split leaks.
* Targets are frequency-floored (`--min-zipf`), because a corpus whose answers
  are words nobody writes measures the dictionary, not the language.
"""

import argparse
import json
import os
import pathlib
import random
import re
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: Bands, by which pool the wrong options came from.
DISTANT, RELATED, SIBLING, SENSE, REJECTED = 1, 2, 3, 4, 5

#: A question is only worth asking if enough wrong options exist to make it one.
MIN_OPTIONS, MAX_OPTIONS = 3, 9

#: A definition shorter than this does not pick a word out of a list.
MIN_GLOSS = 20

#: Sentence length window for the cloze pool. Shorter than 8 words rarely
#: constrains the blank; past 22 the context read dominates the question.
MIN_WORDS, MAX_WORDS = 8, 22

BLANK = "_____"
REJECT = ["none of these", "no option is correct", "none of the above"]

#: How often a reject phrase joins the options, the same whether or not it is
#: the right answer — a phrase that only appears where it is correct is learned
#: as a string instead of as a decision.
REJECT_RATE = 0.2

WORD_RE = re.compile(r"[a-z]+")

PHRASING = {
    "cloze": ["which word fills the blank?",
              "what belongs in the blank?",
              "the blank holds which word?"],
    "usage": ["which sentence uses '{w}' correctly?",
              "in which sentence does '{w}' fit?"],
    "gloss": ["what does '{w}' mean?",
              "which definition belongs to '{w}'?"],
    "name":  ["which word does that describe?",
              "what is being defined?"],
    "kind":  ["what kind of thing is a {w}?",
              "which group does a {w} belong to?"],
    "same":  ["which word means the same as '{w}'?",
              "what is another word for '{w}'?"],
    "opposite": ["which word means the opposite of '{w}'?",
                 "what is the opposite of '{w}'?"],
}


def need_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        from wordfreq import zipf_frequency
    except ImportError as e:
        raise SystemExit(
            f"\n✋ {e.name} is missing.\n"
            f"   pip install nltk wordfreq\n"
            f"   python -c \"import nltk; nltk.download('wordnet')\"\n") from e
    try:
        wn.synsets("test")
    except LookupError as e:
        raise SystemExit(
            "\n✋ WordNet data is missing.\n"
            "   python -c \"import nltk; nltk.download('wordnet')\"\n") from e
    return wn, zipf_frequency


# --------------------------------------------------------------------------- #
# Dictionary: what is askable about which word                                #
# --------------------------------------------------------------------------- #

def single_word(lemma):
    name = lemma.name()
    return "_" not in name and name.isalpha()


class Dictionary:
    """
    WordNet reduced to the questions it can actually answer unambiguously.

    Built once and shared by every split, because the expensive parts — the
    sense index and the frequency lookup over 77k lemmas — do not depend on
    which synsets a split owns.
    """

    def __init__(self, min_zipf, verbose=True):
        wn, zipf = need_wordnet()
        self.wn = wn
        self.synsets = list(wn.all_synsets())

        self.senses = defaultdict(list)
        spellings = defaultdict(set)
        for s in self.synsets:
            for l in s.lemmas():
                if single_word(l):
                    self.senses[l.name().lower()].append(s)
                    spellings[l.name().lower()].add(l.name())

        self.zipf = {w: zipf(w, "en") for w in self.senses}
        self.target = {w for w, v in self.zipf.items()
                       if v >= min_zipf and self.askable(w, spellings[w])}
        self.mono = {w for w in self.target if len(self.senses[w]) == 1}

        # Frequency bands for RELATED distractors, keyed by part of speech.
        # Adjective satellites ('s') answer the same questions as adjectives.
        self.by_pos_band = defaultdict(list)
        for w in self.target:
            pos = self.senses[w][0].pos()
            self.by_pos_band[("a" if pos == "s" else pos,
                              int(self.zipf[w]))].append(w)
        for pool in self.by_pos_band.values():
            pool.sort()

        self.all_targets = sorted(self.target)
        if verbose:
            print(f"📖 WordNet: {len(self.synsets):,} synsets | "
                  f"{len(self.senses):,} single-word lemmas | "
                  f"{len(self.target):,} above zipf {min_zipf} "
                  f"({len(self.mono):,} monosemous)")

    def askable(self, word, spellings):
        """
        Whether a word's meaning is worth asking about.

        Three kinds of lemma are in WordNet but are not vocabulary: proper
        nouns (every spelling capitalised), named entities (every sense an
        instance of something), and abbreviations. Asking which city the
        definition describes tests geography, and `SW` tests nothing — both
        would read as word knowledge in the accuracy.
        """
        if len(word) < 4:
            return False
        if all(s[0].isupper() for s in spellings):
            return False
        return not all(s.instance_hypernyms() for s in self.senses[word])

    def lemmas_of(self, synset):
        return [l.name().lower() for l in synset.lemmas() if single_word(l)]

    def targets_of(self, synset):
        return [w for w in self.lemmas_of(synset) if w in self.target]

    def siblings(self, synset):
        """Words under the same immediate hypernym — the sharpest distractors."""
        out = set()
        for h in synset.hypernyms():
            for child in h.hyponyms():
                if child != synset:
                    out.update(self.targets_of(child))
        return sorted(out)

    def related(self, word):
        """Same part of speech, same frequency band."""
        pos = self.senses[word][0].pos()
        return self.by_pos_band.get(("a" if pos == "s" else pos,
                                     int(self.zipf[word])), ())

    def clean_gloss(self, synset):
        """
        The definition, or None when it cannot stand as a question.

        Two ways a gloss fails: it spells one of its own lemmas out, which
        makes the question a string match and happens thousands of times in
        WordNet ("abaxial: facing away from the axis"); or it is too short to
        pick a word out of a list by ("a Scottish word").
        """
        gloss = synset.definition()
        if not gloss or len(gloss) < MIN_GLOSS:
            return None
        low = gloss.lower()
        for form in (l.name().replace("_", " ").lower() for l in synset.lemmas()):
            if re.search(r"\b" + re.escape(form) + r"\b", low):
                return None
        return gloss


# --------------------------------------------------------------------------- #
# Sentences: real usage, indexed by the words it can teach                    #
# --------------------------------------------------------------------------- #

def load_sentences(path, dictionary, cap, limit, verbose=True):
    """
    {word: [sentence, ...]} over a plain-text, one-sentence-per-line corpus.

    Only words the dictionary can target are indexed and only `cap` sentences
    are kept per word, so the index stays a few tens of MB whatever the file
    weighs — the corpus here is 892 MB and nothing holds more than one line of
    it at a time.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f"\n✋ No sentence corpus at {path}.\n")

    want = dictionary.target
    kept = defaultdict(list)
    lines = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            if limit and lines > limit:
                break
            sentence = line.strip()
            words = sentence.split()
            if not MIN_WORDS <= len(words) <= MAX_WORDS:
                continue
            for word in set(WORD_RE.findall(sentence.lower())):
                if word in want and len(kept[word]) < cap:
                    kept[word].append(sentence)
    usable = {w: s for w, s in kept.items() if len(s) >= 2}
    if verbose:
        print(f"📄 {os.path.basename(path)}: {lines:,} lines | "
              f"{len(usable):,} words with usable sentences")
    return usable


def blank_out(sentence, word):
    """The sentence with every occurrence of `word` replaced, or None."""
    out, n = re.subn(r"\b" + re.escape(word) + r"\b", BLANK, sentence,
                     flags=re.IGNORECASE)
    return out if n else None


# --------------------------------------------------------------------------- #
# Questions                                                                   #
# --------------------------------------------------------------------------- #

class Generator:
    """
    One split's share of the dictionary, as questions.

    A split owns a set of words and the synsets all of whose target lemmas are
    its own, so an answer word in training is never an answer in test — a
    model that memorised the pairing scores at chance here rather than looking
    right until deployment.
    """

    def __init__(self, dictionary, owned, sentences, rng):
        self.d = dictionary
        self.rng = rng
        self.sentences = sentences
        self.owned = owned
        self.synsets = [s for s in dictionary.synsets
                        if dictionary.targets_of(s)
                        and all(w in owned for w in dictionary.targets_of(s))]
        # Only sentences whose blanked word belongs to this split.
        self.cloze_words = sorted(w for w in owned if w in sentences)
        self.mono = [s for s in self.synsets
                     if any(w in self.d.mono for w in self.d.targets_of(s))]
        # Sense contrast needs WordNet's own examples on two senses at once,
        # because a corpus sentence does not say which sense it used.
        self.sense_words = sorted(
            w for w in owned
            if sum(1 for s in self.d.senses[w] if s.examples()) >= 2)

    def mine(self, words):
        """This split's share of `words` — the only ones it may answer with."""
        return [w for w in words if w in self.owned]

    def say(self, kind, **kw):
        return self.rng.choice(PHRASING[kind]).format(**kw)

    def leaks(self, context, options, correct):
        """
        Whether the context hands the answer over.

        A hypernym or antonym question draws its answer from WordNet while its
        context came from a sentence corpus, and the two sometimes meet — "a
        toxin found in the skin of several poison frogs" asked what kind of
        thing a toxin is, with `poison` on the list. One pass over the options
        catches every type of question at once, where a check inside each
        factory would have to be remembered for the next one.

        Only single-word answers are checked: a `usage` option is a whole
        sentence, and the sentence is the question.
        """
        if correct < 0:
            return False
        answer = options[correct]
        if " " in answer:
            return False
        return bool(re.search(r"\b" + re.escape(answer.lower()) + r"\b",
                              context.lower()))

    def options(self, right, pools, extra=()):
        """
        k options including `right`, with the band the wrong ones came from.

        `pools` names a band per distractor source and one of the wide-enough
        ones is drawn at random: taking the first that fits makes every
        question as hard as its target allows, the easy bands never appear,
        and there is nothing to read an accuracy curve against.
        """
        usable = []
        for band, pool in pools:
            wrong = [o for o in pool
                     if o != right and o not in extra and o not in right]
            if len(wrong) + 1 >= MIN_OPTIONS:
                usable.append((band, wrong))
        if not usable:
            return None
        band, wrong = self.rng.choice(usable)
        self.rng.shuffle(wrong)
        k = self.rng.randint(MIN_OPTIONS, min(MAX_OPTIONS, len(wrong) + 1))
        opts = wrong[: k - 1] + [right]
        self.maybe_reject(opts)
        self.rng.shuffle(opts)
        return opts, opts.index(right), band

    def maybe_reject(self, options):
        """
        Sometimes offer a reject phrase as one more option.

        It goes on answerable and unanswerable questions at the same rate, so
        its presence carries no information. Only ever putting it where it is
        correct teaches the string instead of the word, which scores well and
        means nothing.
        """
        if self.rng.random() < REJECT_RATE:
            options.append(self.rng.choice(REJECT))

    def rejection(self, pool, avoid):
        """Options none of which is right — the `correct: -1` branch."""
        wrong = [o for o in pool if o not in avoid]
        if len(wrong) < MIN_OPTIONS:
            return None
        self.rng.shuffle(wrong)
        k = self.rng.randint(MIN_OPTIONS, min(MAX_OPTIONS, len(wrong)))
        opts = wrong[:k]
        self.maybe_reject(opts)
        self.rng.shuffle(opts)
        return opts, -1, REJECTED

    # -- scenes: (context, [question factory, ...]) ------------------------- #

    def scene_cloze(self):
        """A real sentence with a word removed. The strongest signal here."""
        if not self.cloze_words:
            return None
        word = self.rng.choice(self.cloze_words)
        pool = self.sentences[word]
        sentence = self.rng.choice(pool)
        blanked = blank_out(sentence, word)
        if blanked is None:
            return None
        synset = self.d.senses[word][0]

        def q_word():
            got = self.options(word, [
                (SIBLING, self.d.siblings(synset)),
                (RELATED, self.d.related(word)),
                (DISTANT, self.d.all_targets)])
            if not got:
                return None
            opts, correct, band = got
            return self.say("cloze"), opts, correct, band

        def q_kind():
            # "what kind of thing is a X" only parses for a noun, and only a
            # noun has a hypernym worth naming anyway.
            if synset.pos() != "n":
                return None
            hypernyms = synset.hypernyms()
            if len(hypernyms) != 1:
                return None
            # The hypernym is the answer, so it has to be this split's word
            # too — otherwise every split answers "fence" and "jetty" with the
            # same `barrier` and the test vocabulary is the training one.
            names = self.mine(self.d.targets_of(hypernyms[0]))
            if not names:
                return None
            right = names[0]
            got = self.options(right, [
                (RELATED, self.d.related(right)),
                (DISTANT, self.d.all_targets)], extra=(word,))
            if not got:
                return None
            opts, correct, band = got
            return self.say("kind", w=word), opts, correct, band

        return blanked, [q_word, q_kind]

    def scene_usage(self):
        """
        Several sentences, one of which uses the word. Band 2 by construction.

        The question is which context the word belongs in rather than which
        word belongs in a context — the same knowledge read the other way, and
        the only question type here whose options are whole sentences.
        """
        if len(self.cloze_words) < 2:
            return None
        word = self.rng.choice(self.cloze_words)
        if word not in self.d.mono:
            return None
        right = self.rng.choice(self.sentences[word])
        blanked = blank_out(right, word)
        if blanked is None:
            return None

        others = [w for w in self.rng.sample(
            self.cloze_words, min(8, len(self.cloze_words))) if w != word]
        wrong = []
        for other in others:
            for sentence in self.sentences[other]:
                masked = blank_out(sentence, other)
                if masked and BLANK in masked and masked != blanked:
                    wrong.append(masked.replace(BLANK, word))
                    break
        if len(wrong) + 1 < MIN_OPTIONS:
            return None

        k = self.rng.randint(MIN_OPTIONS, min(5, len(wrong) + 1))
        opts = wrong[: k - 1] + [right]
        self.rng.shuffle(opts)
        correct = opts.index(right)

        def q_usage():
            return self.say("usage", w=word), opts, correct, RELATED

        return f"the word is '{word}'", [q_usage]

    def scene_definition(self):
        """A definition, asked both ways round. Monosemous targets only."""
        if not self.mono:
            return None
        synset = self.rng.choice(self.mono)
        gloss = self.d.clean_gloss(synset)
        if gloss is None:
            return None
        words = [w for w in self.d.targets_of(synset) if w in self.d.mono]
        if not words:
            return None
        word = self.rng.choice(words)

        def q_name():
            got = self.options(word, [
                (SIBLING, self.d.siblings(synset)),
                (RELATED, self.d.related(word)),
                (DISTANT, self.d.all_targets)])
            if not got:
                return None
            opts, correct, band = got
            return self.say("name"), opts, correct, band

        def q_same():
            synonyms = self.mine([w for w in self.d.lemmas_of(synset)
                                  if w != word])
            if not synonyms:
                return None
            got = self.options(self.rng.choice(synonyms), [
                (SIBLING, self.d.siblings(synset)),
                (RELATED, self.d.related(word)),
                (DISTANT, self.d.all_targets)], extra=(word,))
            if not got:
                return None
            opts, correct, band = got
            return self.say("same", w=word), opts, correct, band

        def q_opposite():
            antonyms = self.mine(sorted({a.name().lower()
                                         for l in synset.lemmas()
                                         if single_word(l)
                                         for a in l.antonyms()
                                         if single_word(a)}))
            if not antonyms:
                return None
            got = self.options(self.rng.choice(antonyms), [
                (RELATED, self.d.related(word)),
                (DISTANT, self.d.all_targets)], extra=(word,))
            if not got:
                return None
            opts, correct, band = got
            return self.say("opposite", w=word), opts, correct, band

        return gloss, [q_name, q_same, q_opposite]

    def scene_sense(self):
        """
        One sense of a polysemous word against its others. Band 4.

        A bare polysemous word has no single definition, so the context fixes
        the sense by showing it in use — and the distractors are that same
        word's other senses, which no amount of topic matching separates. The
        sentence has to be WordNet's own example, because that is the only
        sentence whose sense is known rather than guessed.
        """
        if not self.sense_words:
            return None
        word = self.rng.choice(self.sense_words)
        senses = [s for s in self.d.senses[word]
                  if s.examples() and self.d.clean_gloss(s)]
        if len(senses) < 2:
            return None
        right = self.rng.choice(senses)
        sentence = self.rng.choice(right.examples())
        others = [self.d.clean_gloss(s) for s in senses if s != right]

        def q_which():
            pool = [g for g in others if g]
            if len(pool) + 1 < MIN_OPTIONS:
                return None
            self.rng.shuffle(pool)
            k = self.rng.randint(MIN_OPTIONS, min(MAX_OPTIONS, len(pool) + 1))
            gloss = self.d.clean_gloss(right)
            opts = pool[: k - 1] + [gloss]
            self.rng.shuffle(opts)
            return (self.say("gloss", w=word), opts, opts.index(gloss), SENSE)

        return sentence, [q_which]

    def scene_unknown(self):
        """
        A definition whose word is not on the list. Band 5.

        The rejection branch the rest of the corpus never reaches, and the
        only one where declining is the answer rather than the least wrong
        pick.
        """
        if not self.mono:
            return None
        synset = self.rng.choice(self.mono)
        gloss = self.d.clean_gloss(synset)
        if gloss is None:
            return None
        forms = self.d.lemmas_of(synset)
        # Siblings are the hard rejection: close enough to be tempting, and
        # none of them is the word being defined. A set would do here, but
        # `related` is indexed by one word and iteration order over a set of
        # strings varies between processes, which would make the whole file
        # depend on the hash seed rather than on --seed.
        pool = self.d.siblings(synset) or list(
            self.d.related(forms[0]) or self.d.all_targets)

        def q_absent():
            got = self.rejection(pool, avoid=forms)
            if not got:
                return None
            opts, correct, band = got
            return self.say("name"), opts, correct, band

        return gloss, [q_absent]


#: How often each scene is drawn. Cloze dominates because a real sentence with
#: a hole in it is the question that most needs the word's meaning, and the
#: dictionary-shaped ones are available for every word rather than only those
#: the sentence corpus covers.
SCENES = (("scene_cloze", 6), ("scene_usage", 2), ("scene_definition", 4),
          ("scene_sense", 2), ("scene_unknown", 2))


def split_words(dictionary, rng):
    """
    Target words divided 80/10/10, so no answer word crosses a split.

    The split is on *words*, not synsets. A synset-level split looks right —
    disjoint senses — but 55% of targets are polysemous and a cloze answer is
    the word itself, so scattering one word's senses puts it on both sides and
    28% of words end up in more than one split. Partitioning words instead
    leaves 90% of synsets entirely inside one split, which is what the scene
    builders ask for.
    """
    words = sorted(dictionary.target)
    rng.shuffle(words)
    n = len(words)
    cut_a, cut_b = int(n * 0.8), int(n * 0.9)
    return {"train": set(words[:cut_a]),
            "val": set(words[cut_a:cut_b]),
            "test": set(words[cut_b:])}


def build(generator, n, rng, max_questions=3):
    """
    Yield `n` rows, so a million-context file never exists in memory at once.

    A scene can fail — its target may have no sibling pool wide enough, or its
    gloss may spell the answer out — and failures are counted rather than
    retried forever, because a budget that cannot be met at all is a bad
    `--min-zipf` and should say so instead of spinning.
    """
    pool = [name for name, weight in SCENES for _ in range(weight)]
    made = misses = 0
    budget = 200 + n * 20
    while made < n:
        scene = getattr(generator, rng.choice(pool))()
        if scene is None:
            misses += 1
            if misses > budget:
                raise SystemExit(
                    f"\n✋ Only {made:,} of {n:,} contexts could be built. "
                    f"Lower --min-zipf or pass --sentences.\n")
            continue
        context, factories = scene
        rng.shuffle(factories)
        questions = []
        for make in factories[:max_questions]:
            got = make()
            if got is None:
                continue
            q, options, correct, band = got
            if generator.leaks(context, options, correct):
                continue
            questions.append({"q": q, "options": options,
                              "correct": correct, "band": band})
        if not questions:
            misses += 1
            continue
        made += 1
        yield {"context": context, "questions": questions}


class Tally:
    """Corpus statistics accumulated as rows stream past, not from a list."""

    def __init__(self):
        self.contexts = self.questions = self.none = 0
        self.k_min, self.k_max = 10 ** 9, 0
        self.bands = defaultdict(int)

    def add(self, row):
        self.contexts += 1
        for q in row["questions"]:
            self.questions += 1
            k = len(q["options"])
            self.k_min, self.k_max = min(self.k_min, k), max(self.k_max, k)
            self.none += q["correct"] < 0
            self.bands[q["band"]] += 1

    def report(self, name):
        print(f"{name}: {self.contexts:,} contexts | {self.questions:,} "
              f"questions ({self.questions/self.contexts:.1f} per context) | "
              f"K {self.k_min}-{self.k_max} | {self.none:,} unanswerable "
              f"({100*self.none/self.questions:.0f}%)")
        print(f"   bands {dict(sorted(self.bands.items()))}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contexts", type=int, default=120000, metavar="N",
                   help="training contexts; val and test get a tenth each "
                        "(default: %(default)s)")
    p.add_argument("--out-dir", default=str(HERE / ".." / "wordnet"),
                   metavar="DIR",
                   help="where the three JSONL files go "
                        "(default: data/wordnet)")
    p.add_argument("--sentences",
                   default=str(REPO / "data" / "wikisent2.txt"), metavar="PATH",
                   help="plain-text sentence corpus for cloze questions, one "
                        "per line; '' disables them (default: %(default)s)")
    p.add_argument("--sentence-lines", type=int, default=0, metavar="N",
                   help="read only the first N lines of it (default: all)")
    p.add_argument("--sentences-per-word", type=int, default=6, metavar="N",
                   help="how many sentences to keep per word, which is what "
                        "bounds the index (default: %(default)s)")
    p.add_argument("--min-zipf", type=float, default=2.5, metavar="Z",
                   help="frequency floor for answer words on the Zipf scale; "
                        "3.0 keeps ~18k words, 2.5 keeps ~28k "
                        "(default: %(default)s)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    out = pathlib.Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dictionary = Dictionary(a.min_zipf)
    sentences = load_sentences(a.sentences or None, dictionary,
                               a.sentences_per_word, a.sentence_lines)
    splits = split_words(dictionary, random.Random(a.seed))

    tenth = max(1, a.contexts // 10)
    plan = (("train", a.contexts), ("val", tenth), ("test", tenth))
    for i, (name, count) in enumerate(plan):
        rng = random.Random(a.seed + i)
        # The sentence index is shared, but each split may only answer with a
        # word it owns, or a test answer would have been seen in training.
        generator = Generator(dictionary, splits[name], sentences, rng)
        path = out / f"{name}.jsonl"
        tally = Tally()
        with open(path, "w", encoding="utf-8") as fh:
            for row in build(generator, count, rng):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                tally.add(row)
        tally.report(path.name)


if __name__ == "__main__":
    sys.exit(main())
