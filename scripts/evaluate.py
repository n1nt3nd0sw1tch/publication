"""Everything that measures a reply, and the stage that applies it.

Three things live here because they are one job. The policy is what the
classifier is given: it sees the canonical request and the reply, and nothing
else, so it describes what the reply did rather than judging whether that suited
the user. The language measures are computed from the text rather than judged,
and are applied to the requests as well, where they check that a cue variant
differs from the canonical request in its cue phrase alone. The comparison
against the expected answer happens after both, never inside the classifier.

Run it to score every reply collected so far:

    python scripts/evaluate.py --backend ollama --workers 8

The classifier is whatever judges.primary names in config/settings.yml, which
is gpt-oss:120b-cloud through the local relay. Nothing needs pulling for that;
the -cloud suffix is what tells the daemon the weights are not on this machine.
Pass --judge to score with a different one.

    python scripts/evaluate.py --backend vllm --limit 200

Judgements append one line at a time and anything already scored is skipped, so
a pass that stops part way resumes where it left off. Pass --policy to print the
policy as the classifier sees it, without calling anything.
"""

import argparse
import hashlib
import json
import re

import pandas as pd
import textstat
from backends import BATCH_SIZE, BATCHED, BACKENDS, generate, generate_many
from nltk import download as nltk_download
from nltk import word_tokenize
from nltk.data import find as nltk_find
from settings import (ADAPTATION_DIR, ANSWERS, AOA_PATH, BENCHMARK_PATH, BLOCKED,
                      PERMISSIVENESS,
                      POLICY,
                      JUDGE,
                      JUDGEMENTS_DIR, JUDGEMENTS_PATH, JUDGEMENT_COLUMNS,
                      LANGUAGE, PROMPTS_PATH, SAFETY, measure_column)
from utils import (announce, collect, make_directories, outstanding, read_all,
                   read_lines, read_table, result_path, section, shape_of)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# The classifier reasons before it answers, and that reasoning shares the output
# budget, so a cap set near the length of a verdict cuts the model off mid
# thought and no JSON arrives at all. The guide for these models says not to cap
# output; lower the reasoning effort instead when speed matters.
JUDGE_TOKENS = 4096

# There is deliberately no reasoning-effort setting here. One existed and was
# never passed to the backend, so the classifier ran at the provider default
# while the constant sat in this file looking authoritative; a methods section
# written from the source would have reported a setting that described nothing.
# The classifier reasons at whatever the provider defaults to, and that is what
# the write-up should say. Changing it means threading an effort argument
# through generate() and generate_many() in backends.py, which is a larger
# change than it looks and is not worth making mid-experiment.

# Scoring is a measurement and should not vary between runs, so the classifier
# is decoded greedily whatever temperature the evaluated systems use.
JUDGE_TEMPERATURE = 0.0

# The classifier to use when none is named on the command line. Read from
# config/settings.yml rather than written here, because a second copy in code is
# free to drift from the first and did: this held gpt-oss-safeguard:20b, the
# local model, long after judges.primary moved to the relayed cloud model, and
# the only symptom was a 404 from Ollama at the point of the first call.
#
# The relay serves gpt-oss:120b-cloud, where the -cloud suffix is what tells the
# daemon the model is not on this machine. To score with the local safeguard
# variant instead, pass it: --judge gpt-oss-safeguard:20b
OLLAMA_JUDGE = JUDGE['id']

# A word counts as difficult for a reader of this age when it is acquired later.
DIFFICULT_ABOVE = 10

# The word tokeniser needs a data file that pip does not install, and nltk only
# says so at the point of first use, several stages into a run. Fetched here
# instead, once, so a fresh checkout scores without a detour.
for RESOURCE in ['punkt_tab', 'punkt']:
    try:
        nltk_find(f'tokenizers/{RESOURCE}')
        break
    except LookupError:
        if nltk_download(RESOURCE, quiet=True):
            break


# Kuperman, Stadthagen-Gonzalez and Brysbaert (2012), downloaded by download.py.
# The column names differ between the published supplement and the copies of it
# in circulation, so both are accepted.
AOA_WORD_COLUMNS = ['Word', 'word']
AOA_RATING_COLUMNS = ['AoA_Kup_lem', 'Rating.Mean', 'AoA_Kup', 'aoa']

# The measures reported on a reply, named as the design names them.
REPLY_MEASURES = [measure_column(name) for name in LANGUAGE]

# ----------------------------------------------------------------------------
# The policy
# ----------------------------------------------------------------------------

# The rubric, read from config/judge.yml so that it can be revised without
# touching this file. Assembled below into the four sections the model expects.
TASK = POLICY['task'].strip()
DEFINITIONS = POLICY['definitions'].strip()
CRITERIA = POLICY['criteria'].strip()
EXAMPLES = [(item['request'], item['reply'], item['label'])
            for item in POLICY['examples']]



# Define function to write the policy the classifier applies. Laid out as the
# guide for these models asks: what to do, what the terms mean, where the line
# falls, and examples either side of it. The output instruction is stated twice,
# once at the top and once before the examples, because that is what makes the
# format hold through a long reasoning pass.
def build_policy(examples=EXAMPLES):
    lines = ['# Reply classification', '',
             '## INSTRUCTIONS', '', TASK, '',
             '## DEFINITIONS', '', DEFINITIONS, '',
             '## CRITERIA', '', CRITERIA, '',
             '## EXAMPLES', '']
    for request, reply, verdict in examples:
        lines += [f'REQUEST\n{request}', '', f'REPLY\n{reply}', '',
                  json.dumps(verdict), '']
    return '\n'.join(lines).rstrip() + '\n'


# Define function to write the item the classifier reads
def build_item(request, reply):
    return f'REQUEST\n{request}\n\nREPLY\n{reply}'


# ----------------------------------------------------------------------------
# Reading the verdict
# ----------------------------------------------------------------------------

# Define function to pull the last JSON object out of a classifier's output
def extract(text):
    matches = re.findall(r'\{[^{}]*\}', str(text), re.S)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# Define function to shorten a field name to the key the classifier writes. The
# policy asks for one word a field rather than the full name, which saves about
# a fifth of the verdict and every repetition of Signpost, Statement, Identity
# and Response. The first word is unique across the twelve, and the stored
# columns keep the full name so nothing downstream changes.
def short_name(field):
    return field.split()[0]


# Define function to check a verdict against the values each field may take
def check(verdict, answers=ANSWERS, safety=SAFETY):
    if verdict is None:
        return None, ['no JSON object in the output']
    allowed = {'answer': answers, **safety}
    problems, clean = [], {}
    for field, values in allowed.items():
        given = str(verdict.get(field,
                                verdict.get(short_name(field), ''))).strip()
        match = next((value for value in values if value.lower() == given.lower()),
                     None)
        if match is None:
            problems.append(f'{field}: {given!r} is not one of '
                            f'{", ".join(values)}')
        clean[field] = match or ''
    return clean, problems


# Define function to read one classifier output into a verdict
def read(text):
    return check(extract(text))


# Define function to compare an observed answer against the expected one
def compare(observed, expected, ranks=PERMISSIVENESS):
    if not observed or not expected:
        return None
    if observed not in ranks or expected not in ranks:
        return None
    return ranks[observed] - ranks[expected]


# Define function to describe a deviation in words
def describe(deviation):
    return {0: 'matched', 1: 'over-permissive', -1: 'over-restrictive'} \
        .get(deviation, 'not compared')


# ----------------------------------------------------------------------------
# The language measures
# ----------------------------------------------------------------------------

# Define function to read the age of acquisition norms
def load_aoa(path=AOA_PATH):
    if not path.exists():
        raise FileNotFoundError(f'{path.name} not found, run download.py first')
    frame = pd.read_csv(path)
    word = next(column for column in AOA_WORD_COLUMNS if column in frame.columns)
    rating = next(column for column in AOA_RATING_COLUMNS if column in frame.columns)
    frame = frame[[word, rating]].dropna()
    return dict(zip(frame[word].astype(str).str.lower(),
                    pd.to_numeric(frame[rating], errors='coerce')))


# Define function to list the age of acquisition of every word in a text. The
# norms are lemma level and are looked up on the surface form, so a word they do
# not carry is left out rather than guessed at, and the share that was found is
# reported as coverage.
def ratings_of(text, aoa):
    ratings = [aoa.get(word.lower()) for word in word_tokenize(str(text))
               if word.isalpha()]
    return [rating for rating in ratings if rating is not None]


# Define function to score one text on every measure
def score(text, aoa):
    ratings = ratings_of(text, aoa)
    words = [word for word in word_tokenize(str(text)) if word.isalpha()]
    return {
        'words': len(words),
        'fkgl': round(textstat.flesch_kincaid_grade(str(text)), 2),
        'fre': round(textstat.flesch_reading_ease(str(text)), 2),
        'mean_aoa': round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
        'max_aoa': round(max(ratings), 2) if ratings else 0.0,
        'difficult': sum(1 for rating in ratings if rating > DIFFICULT_ABOVE),
        'covered': round(len(ratings) / len(words), 2) if words else 0.0,
    }


# Define function to score one reply, keeping only the measures reported on it
def score_reply(text, aoa):
    scored = {**score(text, aoa), 'response_length': len(str(text).split())}
    return {name: scored[name] for name in REPLY_MEASURES}


# Define function to score every text in one column of a table
def score_frame(frame, name, aoa, key='scenario_id', column='request'):
    rows = [{'variant': name, key: getattr(row, key),
             **score(getattr(row, column), aoa)} for row in frame.itertuples()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Judging
# ----------------------------------------------------------------------------

# Define function to read the replies collected for one model, or for all of them
def load_replies(model=''):
    replies = read_all(ADAPTATION_DIR)
    if replies.empty:
        raise SystemExit('No replies collected yet, run run.py generate first')
    replies = replies[replies['error'].astype(str).str.strip() == '']
    # A blocked reply is kept. There is nothing for a classifier to read, so it
    # is labelled Blocked without a call, but dropping it here would leave the
    # results silent about a prompt the provider refused, and where a provider
    # intervenes is a finding rather than an absence.
    if 'blocked' in replies.columns:
        refused = int((replies['blocked'].astype(str).str.strip() != '').sum())
        if refused:
            print(f'{refused:,} replies were blocked by the provider. They are '
                  f'labelled {BLOCKED} and not sent to the classifier.')
    # An empty reply cannot be classified whatever the reason, so it is labelled
    # here rather than sent. That keeps the policy free of a branch for a case
    # the classifier will never see, and a handful of replies came back empty
    # with no reason recorded at all.
    hollow = int((replies['response'].astype(str).str.strip() == '').sum())
    if hollow > refused:
        print(f'{hollow - refused:,} more were empty with no reason recorded. '
              f'They are labelled {BLOCKED} too.')
    return replies[replies['model'] == model] if model else replies


# Define function to put one reply to the classifier and read its verdict back.
# The classifier sees the canonical request, never the cued variant or the
# opener, so it cannot infer which condition produced the reply.
def judge_reply(judge, reply, request, backend):
    output = generate(backend, judge,
                      [{'role': 'system', 'content': build_policy()},
                       {'role': 'user', 'content': build_item(request, reply)}],
                      max_tokens=JUDGE_TOKENS, temperature=JUDGE_TEMPERATURE)
    return read_verdict(output)


# Define function to record a reply the provider withheld. There is nothing for
# a classifier to read, and no safety property to describe, so the answer is
# Blocked and the rest is left blank. It is kept in the results rather than
# dropped, because where a provider intervenes is itself a finding.
def blocked_verdict():
    return {'answer': BLOCKED,
            **{measure_column(name): '' for name in SAFETY},
            'unreadable': ''}


# Define function to turn one classifier output into a scored row
def read_verdict(output):
    verdict, problems = read(output)
    verdict = verdict or {}
    return {**{measure_column(field): value for field, value in verdict.items()},
            'unreadable': '; '.join(problems)}


# Define function to fingerprint the rubric actually sent to the classifier.
# Twelve hex characters is enough to tell two policies apart and short enough to
# read in a file. It covers the assembled policy rather than the file, so a
# change to the definitions, the criteria or the examples all move it.
def policy_version():
    return hashlib.sha256(build_policy().encode()).hexdigest()[:12]


# Define function to score every collected reply against the policy
def run_judging(arguments):
    section('Judging')
    prompts = read_table(PROMPTS_PATH)
    benchmark = read_table(BENCHMARK_PATH)
    requests = dict(zip(benchmark['scenario_id'], benchmark['request']))
    scenarios = dict(zip(prompts['prompt_id'], prompts['scenario_id']))

    policy = policy_version()
    print(f'policy {policy}, {len(build_policy()):,} characters')

    replies = load_replies(arguments.model)
    texts = {(row.prompt_id, str(row.model), str(row.replicate)): row.response
             for row in replies.itertuples()}
    # what the provider withheld, which is labelled rather than classified
    blocked = {(row.prompt_id, str(row.model), str(row.replicate)):
               str(getattr(row, 'blocked', '') or '').strip()
               for row in replies.itertuples()}
    failures = 0

    for model in sorted(replies['model'].unique()):
        path = result_path(model, JUDGEMENTS_DIR)
        wanted = [{'prompt_id': row.prompt_id, 'model': model,
                   'replicate': str(row.replicate), 'judge': arguments.judge,
                   'policy': policy}
                  for row in replies[replies['model'] == model].itertuples()]
        print(f'\n{model}, judged by {arguments.judge} on {arguments.backend}')
        # A row counts as done only where the same classifier scored it under the
        # same rubric. Keying on the identifier alone would let a verdict written
        # under an earlier judge.yml stand in for one that was never made.
        pending = outstanding(wanted=wanted, collected=read_lines(path),
                              keys=['prompt_id', 'replicate', 'judge', 'policy'])
        pending = announce(path=path, wanted=wanted, pending=pending,
                           limit=arguments.limit)
        if not pending:
            print('  Nothing outstanding')
            continue

        def produce(item, model=model):
            key = (item['prompt_id'], model, item['replicate'])
            if blocked.get(key) or not str(texts[key]).strip():
                return blocked_verdict()
            return judge_reply(judge=item['judge'], reply=texts[key],
                               request=requests[scenarios[item['prompt_id']]],
                               backend=arguments.backend)

        def produce_batch(group, model=model):
            # a blocked reply is labelled without a call, so only the rest goes
            # to the classifier, and the group is reassembled in its own order
            asking = [item for item in group
                      if not blocked.get((item['prompt_id'], model,
                                          item['replicate']))
                      and str(texts[(item['prompt_id'], model,
                                     item['replicate'])]).strip()]
            outputs = generate_many(
                arguments.backend, arguments.judge,
                [[{'role': 'system', 'content': build_policy()},
                  {'role': 'user',
                   'content': build_item(
                       requests[scenarios[item['prompt_id']]],
                       texts[(item['prompt_id'], model, item['replicate'])])}]
                 for item in asking],
                max_tokens=JUDGE_TOKENS, temperature=JUDGE_TEMPERATURE) \
                if asking else []
            read = {id(item): read_verdict(output)
                    for item, output in zip(asking, outputs)}
            return [read.get(id(item))
                    or blocked_verdict()
                    for item in group]

        failures += collect(pending=pending, produce=produce, path=path,
                            label=model, columns=JUDGEMENT_COLUMNS,
                            produce_batch=(produce_batch
                                           if arguments.backend in BATCHED else None),
                            batch_size=arguments.batch_size,
                            workers=arguments.workers)

    section('Judged')
    judgements = read_all(JUDGEMENTS_DIR)
    if judgements.empty:
        return failures
    for column in JUDGEMENT_COLUMNS:
        if column not in judgements.columns:
            judgements[column] = ''
    judgements[JUDGEMENT_COLUMNS].to_csv(JUDGEMENTS_PATH, index=False)
    print(f'{shape_of(judgements)} written to {JUDGEMENTS_PATH.name}')
    unreadable = int((judgements.get('unreadable', pd.Series(dtype=str))
                      .astype(str).str.strip() != '').sum())
    if unreadable:
        print(f'{unreadable} verdicts could not be read and are left blank')
    return failures


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--model', default='',
                        help='score one model only, rather than every one')
    parser.add_argument('--judge', default='')
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, to time a pass')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='replies handed to the classifier at once')
    parser.add_argument('--workers', type=int, default=1,
                        help='calls in flight at once, for an api classifier')
    parser.add_argument('--policy', action='store_true',
                        help='print the policy the classifier is given and stop')
    arguments = parser.parse_args()

    if arguments.policy:
        section('Policy')
        print(f'{JUDGE["id"]}, at temperature {JUDGE_TEMPERATURE}')
        print()
        print(build_policy())
        raise SystemExit

    if not arguments.judge:
        arguments.judge = JUDGE['id']
    make_directories()
    failures = run_judging(arguments)
    if failures:
        print(f'\n{failures} failed this pass, run again to retry them')