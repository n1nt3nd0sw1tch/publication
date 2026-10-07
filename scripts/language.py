"""Measure readability, vocabulary, and wording in model replies.

Notes
-----
Compute per-reply language metrics and corpus-level wording comparisons.
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


BULLET = re.compile(r'^[ \t]*(?:[-*\u2022\u2013]|\d+[.)])[ \t]+', re.M)
HEADING = re.compile(r'^[ \t]*#{1,6}[ \t]*', re.M)
LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
URL = re.compile(r'https?://\S+|www\.\S+')
EMPHASIS = re.compile(r'(\*{1,3}|_{1,3}|`{1,3})')
EMOJI = re.compile('[\U0001F000-\U0001FAFF\u2190-\u2BFF\uFE0F\u200D]')
SPACES = re.compile(r'[ \t]+')
BLANKS = re.compile(r'\n{2,}')


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


    lines = []
    for line in text.split('\n'):
        line = SPACES.sub(' ', line).strip()
        if not line:
            continue
        if line[-1] not in '.!?:;':
            line += '.'
        lines.append(line)
    return BLANKS.sub('\n', ' '.join(lines)).strip()


FLOOR = 50


DIFFICULT_ABOVE = 10


NEEDS_LENGTH = ['FKGL', 'FRE', 'Gunning Fog', 'ARI', 'SMOG']


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


def measure(text, norms, floor=0, difficult=DIFFICULT_ABOVE):
    scored = measure_text(text, norms, difficult)
    if scored[measure_column('Response Length')] >= floor:
        return scored
    blanked = {measure_column(name) for name in NEEDS_LENGTH}
    return {name: ('' if name in blanked else value)
            for name, value in scored.items()}


MEASURES = [measure_column(name) for name in LANGUAGE]


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


def load_texts(model=''):
    replies = read_all(ADAPTATION_DIR)
    if replies.empty:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')
    if model:
        replies = replies[replies['model'] == model]
    replies = returned_only(replies)


    replies['replicate'] = pd.to_numeric(replies['replicate'], errors='coerce')
    replies = replies[replies['replicate'].notna()]
    replies['replicate'] = replies['replicate'].astype(int)
    parts = replies['prompt_id'].str.split('-')
    replies['scenario_id'] = parts.str[0] + '-' + parts.str[1]
    replies['condition'] = parts.str[2:].str.join('-')
    replies['scenario_type'] = parts.str[1].str[0].map(
        {'h': 'Harmful', 'a': 'Age Restricted', 'r': 'Rights', 'b': 'Benign'})
    return replies


def vectorise(texts, max_features=20000):
    vectoriser = TfidfVectorizer(lowercase=True, stop_words=list(STOPWORDS),
                                 max_features=max_features, sublinear_tf=True)
    return vectoriser.fit_transform([str(t) for t in texts]), vectoriser


def _paired_cosine(matrix, left, right):
    import numpy as np
    a, b = matrix[left], matrix[right]
    numerator = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    norms = (np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
             * np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel()))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(norms > 0, numerator / norms, np.nan)


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


def returned_only(replies):
    kept = replies['response'].fillna('').astype(str).str.strip().ne('')
    if 'error' in replies.columns:
        kept &= replies['error'].fillna('').astype(str).str.strip().eq('')
    return replies[kept].copy()


def load(model=''):
    frame = read_all(LANGUAGE_DIR)
    if frame.empty:
        raise SystemExit(f'Nothing measured in {LANGUAGE_DIR}, run this first')
    if model:
        frame = frame[frame['model'] == model]
    for column in MEASURES:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')


    frame['replicate'] = pd.to_numeric(frame['replicate'], errors='coerce')
    frame = frame[frame['replicate'].notna()]
    frame['replicate'] = frame['replicate'].astype(int)


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


    frame['signal'] = frame['condition'].map(
        lambda c: 'none' if c == 'neutral'
        else ('stated' if str(c).startswith('age') else 'cue'))
    frame['age'] = pd.to_numeric(
        frame['condition'].str.extract(r'^age(\d+)$')[0], errors='coerce')


    frame['target_grade'] = frame['age'].map(target_grade)
    frame['aae'] = (frame[measure_column('FKGL')]
                    - frame['target_grade']).abs()
    return frame


GRADE_OFFSET = 5
GRADE_CEILING = 12
ADULT_AGE = 18


COARSE_GRADE = [(5, 0.5), (8, 2.0), (11, 5.0), (13, 7.5)]


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
