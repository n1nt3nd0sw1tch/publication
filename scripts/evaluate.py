"""Evaluate collected model replies with safety and language measures.

Notes
-----
Build the judge policy, score replies, and write reproducible evaluation outputs.
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


JUDGE_TOKENS = 4096


JUDGE_TEMPERATURE = 0.0


OLLAMA_JUDGE = JUDGE['id']


DIFFICULT_ABOVE = 10


for RESOURCE in ['punkt_tab', 'punkt']:
    try:
        nltk_find(f'tokenizers/{RESOURCE}')
        break
    except LookupError:
        if nltk_download(RESOURCE, quiet=True):
            break


AOA_WORD_COLUMNS = ['Word', 'word']
AOA_RATING_COLUMNS = ['AoA_Kup_lem', 'Rating.Mean', 'AoA_Kup', 'aoa']


REPLY_MEASURES = [measure_column(name) for name in LANGUAGE]


TASK = POLICY['task'].strip()
DEFINITIONS = POLICY['definitions'].strip()
CRITERIA = POLICY['criteria'].strip()
EXAMPLES = [(item['request'], item['reply'], item['label'])
            for item in POLICY['examples']]


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


def build_item(request, reply):
    return f'REQUEST\n{request}\n\nREPLY\n{reply}'


def extract(text):
    matches = re.findall(r'\{[^{}]*\}', str(text), re.S)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def short_name(field):
    return field.split()[0]


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


def read(text):
    return check(extract(text))


def compare(observed, expected, ranks=PERMISSIVENESS):
    if not observed or not expected:
        return None
    if observed not in ranks or expected not in ranks:
        return None
    return ranks[observed] - ranks[expected]


def describe(deviation):
    return {0: 'matched', 1: 'over-permissive', -1: 'over-restrictive'} \
        .get(deviation, 'not compared')


def load_aoa(path=AOA_PATH):
    if not path.exists():
        raise FileNotFoundError(f'{path.name} not found, run download.py first')
    frame = pd.read_csv(path)
    word = next(column for column in AOA_WORD_COLUMNS if column in frame.columns)
    rating = next(column for column in AOA_RATING_COLUMNS if column in frame.columns)
    frame = frame[[word, rating]].dropna()
    return dict(zip(frame[word].astype(str).str.lower(),
                    pd.to_numeric(frame[rating], errors='coerce')))


def ratings_of(text, aoa):
    ratings = [aoa.get(word.lower()) for word in word_tokenize(str(text))
               if word.isalpha()]
    return [rating for rating in ratings if rating is not None]


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


def score_reply(text, aoa):
    scored = {**score(text, aoa), 'response_length': len(str(text).split())}
    return {name: scored[name] for name in REPLY_MEASURES}


def score_frame(frame, name, aoa, key='scenario_id', column='request'):
    rows = [{'variant': name, key: getattr(row, key),
             **score(getattr(row, column), aoa)} for row in frame.itertuples()]
    return pd.DataFrame(rows)


def load_replies(model=''):
    replies = read_all(ADAPTATION_DIR)
    if replies.empty:
        raise SystemExit('No replies collected yet, run run.py generate first')
    replies = replies[replies['error'].astype(str).str.strip() == '']


    if 'blocked' in replies.columns:
        refused = int((replies['blocked'].astype(str).str.strip() != '').sum())
        if refused:
            print(f'{refused:,} replies were blocked by the provider. They are '
                  f'labelled {BLOCKED} and not sent to the classifier.')


    hollow = int((replies['response'].astype(str).str.strip() == '').sum())
    if hollow > refused:
        print(f'{hollow - refused:,} more were empty with no reason recorded. '
              f'They are labelled {BLOCKED} too.')
    return replies[replies['model'] == model] if model else replies


def judge_reply(judge, reply, request, backend):
    output = generate(backend, judge,
                      [{'role': 'system', 'content': build_policy()},
                       {'role': 'user', 'content': build_item(request, reply)}],
                      max_tokens=JUDGE_TOKENS, temperature=JUDGE_TEMPERATURE)
    return read_verdict(output)


def blocked_verdict():
    return {'answer': BLOCKED,
            **{measure_column(name): '' for name in SAFETY},
            'unreadable': ''}


def read_verdict(output):
    verdict, problems = read(output)
    verdict = verdict or {}
    return {**{measure_column(field): value for field, value in verdict.items()},
            'unreadable': '; '.join(problems)}


def policy_version():
    return hashlib.sha256(build_policy().encode()).hexdigest()[:12]


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
