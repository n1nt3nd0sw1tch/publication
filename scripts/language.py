"""Measures the language of every collected reply.

    python scripts/language.py
    python scripts/language.py --model claude-haiku-4-5-20251001
    python scripts/language.py --floor 0 --difficult 10

Reads results/adaptation/ and writes results/language/, one file per model.

These are computed from the text and need no model, which is why they are here
rather than in the judgement. A classification costs a call to a classifier and
cannot be revised without another one; readability costs arithmetic, so a change
of mind about how it is measured is a second rather than an afternoon.

Fifteen measures in three groups.

    Readability, how hard the reply is to read
      FKGL              Flesch Kincaid grade level, US school year
      FRE               Flesch reading ease, 0 to 100, higher is easier
      Gunning Fog       years of education implied, weights long words
      ARI               automated readability index, characters not syllables
      SMOG              grade implied by polysyllable density

    Vocabulary, how early the words are learned
      Mean AoA          mean age of acquisition over tokens in the norms
      P90 AoA           the ninetieth percentile of the same distribution
      Max AoA           the latest-acquired token in the reply
      Difficult Share   share of rated tokens acquired after the threshold age
      AoA Coverage      share of words the norms carry at all

    Structure, what the reply is made of
      Response Length   whitespace-separated words
      Sentence Length   words per sentence
      Word Length       characters per word
      TTR               distinct words over total words
      MTLD              measure of textual lexical diversity

Two corpus-level comparisons live here too, since they are arithmetic over the
same text and belong beside the per-reply measures rather than in a notebook.
Cosine similarity asks how far the wording moves between conditions, and the
distinctive-word scores ask which words carry that movement. Both are described
where they are defined below.

Five formulas rather than two because they disagree in informative ways. FKGL
and FRE are two readings of the same two inputs, sentence length and syllables a
word, so they move together by construction and neither adds anything to the
other. Gunning Fog counts long words instead of syllables, ARI counts characters
instead of syllables, and SMOG counts only polysyllables, so a reply that is
simple by one and hard by another is telling you which property of the text is
doing the work. Reporting one of each family is the reason for the set.

A floor applies to the readability group, and it is an analysis decision rather
than a measurement one. Every measure here is computed without a length
restriction, and the analysis then treats the five formula measures as missing
below fifty words, so the threshold can be varied without measuring again.

That floor applies to those five and to nothing else. The vocabulary measures
and most of the structural ones are defined at any length and are written
whatever a reply's length. MTLD is the one exception, and its limit is its own
rather than the analysis floor: it needs fifty tokens to run at all and returns
nothing below that.

Readability formulas are unstable on
short texts and meaningless on very short ones: 'I cannot help with that' has a
grade level below zero, which is arithmetic rather than a finding. Replies under
the floor carry the structure and vocabulary measures, which are defined at any
length, and are left blank for the five formulas. The share left blank is
reported per model, since it differs enormously between them and is itself a
result: a model that refuses in six words is not missing data, it is refusing in
six words.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import textstat
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import evaluate
from settings import (ADAPTATION_DIR, BENCHMARK_PATH, LANGUAGE, LANGUAGE_COLUMNS,
                      LANGUAGE_DIR, PROMPTS_PATH, measure_column)
from utils import (append_line, make_directories, read_all, read_lines,
                   read_table, result_path, section)

# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------

# Readability formulas divide by the sentence count, and a markdown list has no
# terminal punctuation, so a reply written as bullets is read as one enormous
# sentence. In this corpus that is not a rare accident: it puts 5% of measured
# replies above grade 20 and almost all of them belong to the model that uses
# lists most. Measuring the raw text therefore compares formatting habits rather
# than prose, so every measure is computed on a cleaned field and the raw text is
# kept beside it.
BULLET = re.compile(r'^[ \t]*(?:[-*\u2022\u2013]|\d+[.)])[ \t]+', re.M)
HEADING = re.compile(r'^[ \t]*#{1,6}[ \t]*', re.M)
LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
URL = re.compile(r'https?://\S+|www\.\S+')
EMPHASIS = re.compile(r'(\*{1,3}|_{1,3}|`{1,3})')
EMOJI = re.compile('[\U0001F000-\U0001FAFF\u2190-\u2BFF\uFE0F\u200D]')
SPACES = re.compile(r'[ \t]+')
BLANKS = re.compile(r'\n{2,}')


# Define function to turn a formatted reply into running prose, so that the
# readability formulas measure the sentences rather than the layout
def clean(text):
    text = str(text)
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ', ').replace('\u2013', ', ')
    text = LINK.sub(r'\1', text)
    text = URL.sub(' ', text)
    text = HEADING.sub('', text)
    text = BULLET.sub('', text)
    text = EMPHASIS.sub('', text)
    text = EMOJI.sub(' ', text)
    # A list item is a sentence for counting purposes. Without this the whole
    # list is one sentence and words per sentence runs into the hundreds.
    lines = []
    for line in text.split('\n'):
        line = SPACES.sub(' ', line).strip()
        if not line:
            continue
        if line[-1] not in '.!?:;':
            line += '.'
        lines.append(line)
    return BLANKS.sub('\n', ' '.join(lines)).strip()


# Words below which the readability formulas are not reported. A hundred is the
# usual floor for FKGL to be stable and fifty is where it stops being nonsense;
# fifty is used, so that the measure covers as much of the data as it can while
# still meaning something.
FLOOR = 50

# The age after which a word counts as difficult. Ten sits between the youngest
# and oldest ages the benchmark discloses, so the share above it separates a
# reply pitched at a young child from one pitched at anybody older.
DIFFICULT_ABOVE = 10

# Measures that need a certain length before they mean anything. Everything else
# is defined on a single sentence.
NEEDS_LENGTH = ['FKGL', 'FRE', 'Gunning Fog', 'ARI', 'SMOG']


# Define function to measure lexical diversity in a way that does not fall with
# length. A type-token ratio does: a 500 word reply repeats function words more
# than a 50 word one, so TTR reads as diversity but partly measures brevity, and
# length here moves with the age condition. MTLD instead counts how many tokens
# it takes for the running type-token ratio to fall to a threshold, and averages
# that run length forwards and backwards, which is stable across lengths above
# roughly a hundred tokens.
def mtld(words, threshold=0.72):
    def run(sequence):
        factors, types, tokens = 0.0, set(), 0
        for word in sequence:
            types.add(word); tokens += 1
            if len(types) / tokens <= threshold:
                factors += 1; types, tokens = set(), 0
        if tokens:
            factors += (1 - len(types) / tokens) / (1 - threshold)
        return len(sequence) / factors if factors else None
    if len(words) < 50:
        return None
    forward, backward = run(words), run(words[::-1])
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2


# Define function to compute every measure on one text, before any floor is
# applied. Rounding happens here so that what is written and what is read back
# are the same number.
def measure_text(text, norms, difficult=DIFFICULT_ABOVE):
    text = clean(text)
    words = [word for word in word_tokenize(text) if word.isalpha()]
    ratings = evaluate.ratings_of(text, norms)
    sentences = max(textstat.sentence_count(text), 1)
    lowered = [word.lower() for word in words]

    import numpy as np
    scored = {
        'FKGL': textstat.flesch_kincaid_grade(text),
        'FRE': textstat.flesch_reading_ease(text),
        'Gunning Fog': textstat.gunning_fog(text),
        'ARI': textstat.automated_readability_index(text),
        'SMOG': textstat.smog_index(text),
        'Mean AoA': sum(ratings) / len(ratings) if ratings else None,
        'P90 AoA': float(np.percentile(ratings, 90)) if ratings else None,
        'Max AoA': max(ratings) if ratings else None,
        'Difficult Share': (sum(1 for r in ratings if r > difficult)
                            / len(ratings)) if ratings else None,
        'AoA Coverage': len(ratings) / len(words) if words else None,
        'Response Length': len(text.split()),
        'Sentence Length': len(words) / sentences if words else None,
        'Word Length': sum(len(w) for w in words) / len(words) if words else None,
        'TTR': len(set(lowered)) / len(lowered) if lowered else None,
        'MTLD': mtld(lowered),
    }
    return {measure_column(name): ('' if value is None else round(value, 3))
            for name, value in scored.items()}


# Define function to blank the length-dependent measures on a reply too short
# for them to mean anything, keeping the rest
def measure(text, norms, floor=0, difficult=DIFFICULT_ABOVE):
    scored = measure_text(text, norms, difficult)
    if scored[measure_column('Response Length')] >= floor:
        return scored
    blanked = {measure_column(name) for name in NEEDS_LENGTH}
    return {name: ('' if name in blanked else value)
            for name, value in scored.items()}


MEASURES = [measure_column(name) for name in LANGUAGE]

# ----------------------------------------------------------------------------
# Wording
# ----------------------------------------------------------------------------

# Words too common to distinguish anything. Kept short and explicit rather than
# imported, so that what was removed is visible in the source: a stop list is a
# modelling choice and a silent one is hard to argue with later.
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing don
down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my
myself no nor not of off on once only or other our ours ourselves out over own
re s same she should so some such t than that the their theirs them themselves
then there these they this those through to too under until up very was we were
what when where which while who whom why will with you your yours yourself
yourselves ll ve d m o y ain aren couldn didn doesn hadn hasn haven isn ma
mightn mustn needn shan shouldn wasn weren won wouldn also would may might
""".split())


# Words that survive the stop list but say nothing about who the reply was for.
# Every one of these appeared in the top fifteen of at least one contrast, and
# "specific", "often" and "rather" appeared on the adult side of all four
# scenario types, which is a fact about register rather than about age
# adaptation. They are listed rather than filtered by frequency so that what was
# removed is visible and arguable: a silent stop list cannot be defended later.
GENERIC = set("""
like make makes making really thing things say says said get gets getting got
want wants need needs know knows take takes give gives go goes going come comes
way ways good better best well lot much many also even still back put use used
using actually maybe probably usually generally especially particularly
sometimes often rather specific specifically something someone anything
everything different various certain several provide provides providing without
sure kind able one two first
""".split())
STOPWORDS |= GENERIC


# Define function to read the collected replies with their condition attached,
# since the measures written per reply carry no text and the wording comparisons
# need it
def load_texts(model=''):
    replies = read_all(ADAPTATION_DIR)
    if replies.empty:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')
    if model:
        replies = replies[replies['model'] == model]
    replies = returned_only(replies)
    # The replicate arrives as a string from some collectors and an integer from
    # others, and the pairing below indexes on it, so it is coerced once here
    # rather than compared across types further down.
    replies['replicate'] = pd.to_numeric(replies['replicate'], errors='coerce')
    replies = replies[replies['replicate'].notna()]
    replies['replicate'] = replies['replicate'].astype(int)
    parts = replies['prompt_id'].str.split('-')
    replies['scenario_id'] = parts.str[0] + '-' + parts.str[1]
    replies['condition'] = parts.str[2:].str.join('-')
    replies['scenario_type'] = parts.str[1].str[0].map(
        {'h': 'Harmful', 'a': 'Age Restricted', 'r': 'Rights', 'b': 'Benign'})
    return replies


# Define function to score every reply against a shared vocabulary, so that two
# replies can be compared. Fitting one vectoriser over the whole set rather than
# one per pair is what makes the numbers comparable across pairs.
def vectorise(texts, max_features=20000):
    vectoriser = TfidfVectorizer(lowercase=True, stop_words=list(STOPWORDS),
                                 max_features=max_features, sublinear_tf=True)
    return vectoriser.fit_transform([str(t) for t in texts]), vectoriser


# Define function to average the cosine between paired rows of a matrix
def _paired_cosine(matrix, left, right):
    import numpy as np
    a, b = matrix[left], matrix[right]
    numerator = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    norms = (np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
             * np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel()))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(norms > 0, numerator / norms, np.nan)


# Define function to measure how far the wording moves between two conditions,
# holding the scenario, the model and the replicate fixed.
#
# The number means nothing on its own. Two replies to the same prompt from the
# same model differ because the decoding is stochastic, so a cosine of 0.5
# between two ages is only evidence of adaptation if two draws of one age score
# higher than that. Call replicate_similarity for that floor and report the two
# together; the difference between them is the effect, not the cosine itself.
def condition_similarity(replies, first, second):
    subset = replies[replies['condition'].isin([first, second])]
    matrix, _ = vectorise(subset['response'])
    index = {key: position for position, key in enumerate(
        zip(subset['model'], subset['scenario_id'], subset['replicate'],
            subset['condition']))}
    rows = []
    for (model, scenario, replicate, condition), position in index.items():
        if condition != first:
            continue
        other = index.get((model, scenario, replicate, second))
        if other is None:
            continue
        rows.append({'model': model, 'scenario_id': scenario,
                     'replicate': replicate, 'left': position, 'right': other})
    if not rows:
        return pd.DataFrame(columns=['model', 'scenario_id', 'replicate',
                                     'cosine'])
    pairs = pd.DataFrame(rows)
    pairs['cosine'] = _paired_cosine(matrix, pairs['left'].to_numpy(),
                                     pairs['right'].to_numpy())
    return pairs.drop(columns=['left', 'right'])


# Define function to measure the floor: how far the wording moves between two
# draws of the same prompt, which is variation the condition did not cause
def replicate_similarity(replies, condition):
    subset = replies[replies['condition'] == condition]
    matrix, _ = vectorise(subset['response'])
    index = {key: position for position, key in enumerate(
        zip(subset['model'], subset['scenario_id'], subset['replicate']))}
    rows = []
    for (model, scenario, replicate), position in index.items():
        other = index.get((model, scenario, replicate + 1))
        if other is None:
            continue
        rows.append({'model': model, 'scenario_id': scenario,
                     'left': position, 'right': other})
    if not rows:
        return pd.DataFrame(columns=['model', 'scenario_id', 'cosine'])
    pairs = pd.DataFrame(rows)
    pairs['cosine'] = _paired_cosine(matrix, pairs['left'].to_numpy(),
                                     pairs['right'].to_numpy())
    return pairs.drop(columns=['left', 'right'])


# Define function to score which words distinguish one set of replies from
#
# A plain frequency count answers with 'you', 'the' and 'help' whatever the
# split, and a raw log ratio answers with whatever appeared twice in one set and
# never in the other. The prior is the pooled corpus, so a word is distinctive
# only if it is commoner here than the corpus as a whole would predict, and the
# denominator penalises rare words rather than rewarding them.
# Define function to rank the words that separate two sets of replies
#
# Returns a weighted log-odds z-score, not a ratio. The log odds of a word in
# one set against the other are taken with an informative Dirichlet prior drawn
# from the pooled corpus, and the result is then divided by its estimated
# standard error, which is the last step and the one the name has to record.
#
# Each part does its own work. A plain frequency count answers with 'you', 'the'
# and 'help' whatever the split. A raw log ratio answers with whatever appeared
# twice in one set and never in the other. The prior is what makes a word large
# only where it is commoner than the pooled corpus predicts, and the division by
# the standard error is what stops a word seen a handful of times from
# outranking one seen a thousand.
#
# Say weighted log-odds z-score wherever the measure is named. Calling it a
# ratio describes the numerator and omits the standardisation, which is the part
# that makes the ranking usable.
def distinctive_words(left, right, prior_weight=1000, minimum=15):
    from collections import Counter
    import numpy as np

    def count(texts):
        counter = Counter()
        for text in texts:
            counter.update(word for word in
                           word_tokenize(str(text).lower())
                           if word.isalpha() and word not in STOPWORDS
                           and len(word) > 2)
        return counter

    first, second = count(left), count(right)
    pooled = first + second
    vocabulary = [word for word, total in pooled.items() if total >= minimum]
    size = sum(pooled.values())

    scores = {}
    for word in vocabulary:
        prior = prior_weight * pooled[word] / size
        a = first[word] + prior
        b = second[word] + prior
        odds = (np.log(a / (sum(first.values()) + prior_weight - a))
                - np.log(b / (sum(second.values()) + prior_weight - b)))
        scores[word] = odds / np.sqrt(1 / a + 1 / b)
    return pd.Series(scores).sort_values(ascending=False)




# Define function to keep the replies a provider actually returned
#
# One definition, used by the measuring pass and by load_texts, because two
# copies of it drift and the drift is silent: a corpus measured under one rule
# and read under another gives a denominator that matches nothing.
#
# fillna is load-bearing. A withheld reply can arrive as an empty string or as a
# null, depending on how the provider reported it, and str(NaN) is the three
# characters 'nan', so a filter written as .astype(str).str.strip() != '' keeps
# every null it was written to remove. That failure is silent twice over: the
# rows measure as zero-length replies and land in every denominator, and a check
# that compares the measured count against load_texts passes because both sides
# made the same mistake.
def returned_only(replies):
    kept = replies['response'].fillna('').astype(str).str.strip().ne('')
    if 'error' in replies.columns:
        kept &= replies['error'].fillna('').astype(str).str.strip().eq('')
    return replies[kept].copy()

# Define function to read what was written, as numbers and with the experimental
# metadata attached. Blanked cells are written as empty strings, which makes
# every column object dtype, so a mean over one silently fails; coercing here
# means an analysis never has to remember to. Anything blank becomes NaN and is
# skipped by every pandas aggregation, which is the behaviour wanted: a reply too
# short to measure should not count as a zero.
def load(model=''):
    frame = read_all(LANGUAGE_DIR)
    if frame.empty:
        raise SystemExit(f'Nothing measured in {LANGUAGE_DIR}, run this first')
    if model:
        frame = frame[frame['model'] == model]
    for column in MEASURES:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    # load_texts coerces the replicate and this did not, so the two frames would
    # not join. Coerced in both, once, rather than at every call site.
    frame['replicate'] = pd.to_numeric(frame['replicate'], errors='coerce')
    frame = frame[frame['replicate'].notna()]
    frame['replicate'] = frame['replicate'].astype(int)

    # The condition and the stratum come from the prompt table where it exists,
    # and from the identifier otherwise, so the notebook works on a fresh clone.
    if PROMPTS_PATH.exists() and BENCHMARK_PATH.exists():
        prompts = read_table(PROMPTS_PATH)
        benchmark = read_table(BENCHMARK_PATH)
        facts = prompts[['prompt_id', 'scenario_id', 'condition', 'band']].merge(
            benchmark[['scenario_id', 'domain', 'scenario_type', 'category']],
            on='scenario_id')
        frame = frame.merge(facts, on='prompt_id', how='left')
    else:
        parts = frame['prompt_id'].str.split('-')
        frame['scenario_id'] = parts.str[0] + '-' + parts.str[1]
        frame['condition'] = parts.str[2:].str.join('-')
        frame['scenario_type'] = parts.str[1].str[0].map(
            {'h': 'Harmful', 'a': 'Age Restricted',
             'r': 'Rights', 'b': 'Benign'})
        frame['domain'] = parts.str[0]

    # Signal and age are read off the condition so that the stated arm and the
    # cue arm can be reported apart without a second lookup.
    frame['signal'] = frame['condition'].map(
        lambda c: 'none' if c == 'neutral'
        else ('stated' if str(c).startswith('age') else 'cue'))
    frame['age'] = pd.to_numeric(
        frame['condition'].str.extract(r'^age(\d+)$')[0], errors='coerce')

    # Alignment is computed here rather than per reply, because it needs the age
    # the reply was told and the per-reply pass never sees the condition.
    frame['target_grade'] = frame['age'].map(target_grade)
    frame['aae'] = (frame[measure_column('FKGL')]
                    - frame['target_grade']).abs()
    return frame


# ----------------------------------------------------------------------------
# Age alignment
# ----------------------------------------------------------------------------

# The reading grade a reply would have to hit to match the age it was told. US
# grade level runs about five years behind chronological age, so a seven year
# old sits in grade two, and the scale stops at grade twelve.
#
# It is defined only below eighteen, and deliberately. Grade level is a schooling
# scale, and past the end of schooling there is no grade an adult reply ought to
# hit: an answer to a twenty-one year old is not better for being pitched at
# grade sixteen. Assigning adults a target would invent a standard the scale does
# not carry, so the two adult conditions are reference points and carry no
# alignment error.
GRADE_OFFSET = 5
GRADE_CEILING = 12
ADULT_AGE = 18


# Define function to give the target reading grade for a stated age, or nothing
# where no defensible target exists
# The mapping this measure is taken from, as a step function stopping at
# thirteen. Retained for the sensitivity check: the continuous extension above
# is this thesis's generalisation, and the primary result is reported only where
# both mappings agree on direction. The two overlap on ages 7, 9, 11 and 13, so
# the check covers four of the six stated minor ages and not the whole ladder.
COARSE_GRADE = [(5, 0.5), (8, 2.0), (11, 5.0), (13, 7.5)]


# Define function to give the source's coarse target grade, or nothing above
# the range it covers
def coarse_target_grade(age):
    if age is None or pd.isna(age):
        return None
    for ceiling, grade in COARSE_GRADE:
        if age <= ceiling:
            return grade
    return None


def target_grade(age):
    if age is None or pd.isna(age) or age >= ADULT_AGE:
        return None
    return min(float(age) - GRADE_OFFSET, GRADE_CEILING)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

# Define function to put an interval on a mean by resampling scenarios rather
# than replies.
#
# The 200 scenarios are the sampled unit; the thirteen conditions and three
# replicates within one are not independent of each other. Resampling replies
# would treat 46,800 correlated observations as 46,800 independent ones and give
# intervals several times too narrow. Resampling whole scenarios keeps whatever
# is peculiar to a scenario together, which is what a cluster bootstrap is for.
def bootstrap(frame, column, cluster='scenario_id', draws=1000, seed=7):
    import numpy as np
    frame = frame[[column, cluster]].dropna()
    if frame.empty:
        return None, None, None
    groups = frame.groupby(cluster)[column].agg(['sum', 'count'])
    sums, counts = groups['sum'].to_numpy(), groups['count'].to_numpy()
    if len(sums) < 2:
        return float(frame[column].mean()), None, None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (float(frame[column].mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


# Define function to bootstrap a paired difference, resampling the scenarios
# that carry both sides of the pair
def bootstrap_paired(frame, column, cluster='scenario_id', draws=1000, seed=7):
    import numpy as np
    frame = frame[[column, cluster]].dropna()
    if frame[cluster].nunique() < 2:
        return None, None, None
    per = frame.groupby(cluster)[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per), size=(draws, len(per)))
    means = per[picks].mean(axis=1)
    return (float(per.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='', help='one model, or all of them')
    # Zero, not FLOOR. The frozen pass was measured without a length
    # restriction and notebooks/16_readability.ipynb applies the fifty-word
    # floor itself, which is what lets the threshold be varied in the
    # sensitivity analysis without measuring the corpus again. A default of
    # fifty would let a bare `python scripts/language.py` write files that fail
    # the notebook's assertion that every FKGL is present, on a corpus that had
    # been silently re-measured. FLOOR stays defined because measure() takes it
    # and the notebook imports it as the analysis floor.
    parser.add_argument('--floor', type=int, default=0,
                        help='words below which readability is left blank')
    parser.add_argument('--difficult', type=float, default=DIFFICULT_ABOVE,
                        help='age after which a word counts as difficult')
    arguments = parser.parse_args()

    make_directories()
    norms = evaluate.load_aoa()
    files = ([result_path(arguments.model, ADAPTATION_DIR)] if arguments.model
             else sorted(ADAPTATION_DIR.glob('*.jsonl')))
    if not files:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')

    section('Language')
    print(f'{len(LANGUAGE_COLUMNS) - 3} measures, floor {arguments.floor} words, '
          f'difficult above age {arguments.difficult:g}\n')
    for path in files:
        replies = read_lines(path)
        if replies.empty:
            continue

        # RETURNED REPLIES ONLY. DO NOT REMOVE THIS.
        #
        # A reply the provider withheld is an empty string or a null. Measured,
        # it scores zero words, falls below any floor, and is written out with
        # the five formulas blank, so it lands in every denominator and reports
        # an intervention outside the model as a model writing nothing. It also
        # matches no token in the norms, so it arrives in the AoA missingness
        # count as well: on this corpus that is 160 rows, which turns 6 genuine
        # unmatched replies into 166.
        #
        # The corpus is 46,800 requests and 46,640 replies, and Section 3.4.2
        # excludes a withheld reply from every rate because a provider-level
        # block is a property of the service rather than of the model.
        before = len(replies)
        replies = returned_only(replies)
        withheld = before - len(replies)
        if replies.empty:
            continue

        rows = []
        for reply in replies.itertuples():
            rows.append({'prompt_id': reply.prompt_id, 'model': reply.model,
                         'replicate': reply.replicate,
                         **measure(str(reply.response), norms,
                                   arguments.floor, arguments.difficult)})

        model = str(replies['model'].iloc[0])
        written = result_path(model, LANGUAGE_DIR)
        written.unlink(missing_ok=True)
        for row in rows:
            append_line(written, {name: row.get(name, '')
                                  for name in LANGUAGE_COLUMNS})

        measured = sum(1 for row in rows if row[measure_column('FKGL')] != '')
        lengths = sorted(row[measure_column('Response Length')] for row in rows)
        note = f', {withheld} withheld' if withheld else ''
        print(f'  {model:<28} {len(rows):>6,} replies{note}, {measured:>6,} '
              f'long enough for the formulas ({measured / len(rows):>4.0%}), '
              f'median {lengths[len(lengths) // 2]:,} words')

    print(f'\nWritten to {LANGUAGE_DIR}')
    print(f'Replies under {arguments.floor} words carry vocabulary and structure '
          f'and are blank on the five formulas. Report that share per model: it '
          f'is not attrition, it is how briefly a model refuses.')
