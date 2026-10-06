"""Run the judge on the rows you labelled by hand, and compare.

Uses the same policy, the same classifier and the same request text the full
pass will use: the bare scenario request, not the prompt with the age opener,
so the classifier sees what it will see in production.

    python scripts/agree.py                          every human file
    python scripts/agree.py --model mistral-small-2603
    python scripts/agree.py --workers 8

Reads results/annotation/manual/*_human_labels.csv, writes results/annotation/
agreement.csv and prints the disagreements worth reading.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from evaluate import judge_reply
from settings import (ANNOTATION_DIR, BENCHMARK_PATH, JUDGE, MANUAL_DIR,
                      PROMPTS_PATH, RESULTS_DIR, SAFETY,
                      measure_column)
from utils import read_table, section

# The hand labels live where settings says they live. This was a literal path
# with a space in it, which worked on one machine and quietly found nothing on
# any other.

FIELDS = ['answer'] + [measure_column(name) for name in SAFETY]


# Define function to correct agreement for the share that would happen by
# chance given how often each value occurs
def kappa(left, right):
    pairs = [(a, b) for a, b in zip(left, right)
             if str(a).strip() and str(b).strip()]
    if not pairs:
        return None, 0, None
    agreed = sum(a == b for a, b in pairs) / len(pairs)
    values = {value for pair in pairs for value in pair}
    expected = sum((sum(a == v for a, _ in pairs) / len(pairs))
                   * (sum(b == v for _, b in pairs) / len(pairs))
                   for v in values)
    if expected >= 1:
        return None, len(pairs), agreed
    return round((agreed - expected) / (1 - expected), 3), len(pairs), agreed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='ollama')
    parser.add_argument('--judge', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--workers', type=int, default=8)
    arguments = parser.parse_args()
    if not arguments.judge:
        arguments.judge = JUDGE['id']

    sheets = sorted(MANUAL_DIR.glob('*_human_labels.csv'))
    if arguments.model:
        sheets = [p for p in sheets if p.stem.startswith(arguments.model)]
    if not sheets:
        raise SystemExit(f'No *_human_labels.csv in {MANUAL_DIR}')

    human = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False)
                       for p in sheets], ignore_index=True)
    human = human[human['answer'].str.strip() != '']
    if 'uncertain' in human:
        unsure = int((human['uncertain'] == 'unsure').sum())
        human = human[human['uncertain'] != 'unsure']
    else:
        unsure = 0

    # The bare request, as the classifier gets it in the full pass. The prompt
    # column carries the age opener and is deliberately not used.
    prompts = read_table(PROMPTS_PATH)
    benchmark = read_table(BENCHMARK_PATH)
    scenario = dict(zip(prompts['prompt_id'], prompts['scenario_id']))
    request = dict(zip(benchmark['scenario_id'], benchmark['request']))

    section('Judging')
    print(f'{len(human)} rows, {unsure} dropped as unsure')
    print(f'{arguments.judge} on {arguments.backend}, {len(FIELDS)} fields')

    def score(row):
        return judge_reply(judge=arguments.judge, reply=row.response,
                           request=request[scenario[row.prompt_id]],
                           backend=arguments.backend)

    rows = list(human.itertuples())
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        verdicts = list(pool.map(score, rows))

    machine = pd.DataFrame(verdicts).fillna('')
    for field in FIELDS:
        if field not in machine:
            machine[field] = ''
    machine.index = human.index

    section('Agreement')
    unread = int((machine.get('unreadable', pd.Series('', index=machine.index))
                  .astype(str).str.strip() != '').sum())
    if unread:
        print(f'{unread} verdicts could not be read\n')
    print(f'  {"field":24} {"kappa":>7} {"raw":>6} {"you":>6} {"judge":>6}  n')
    report = []
    for field in FIELDS:
        score_, n, raw = kappa(human[field], machine[field])
        yours = 100 * (human[field] == 'Yes').mean() if field != 'answer' \
            else 100 * (human[field] == 'Compliance').mean()
        theirs = 100 * (machine[field] == 'Yes').mean() if field != 'answer' \
            else 100 * (machine[field] == 'Compliance').mean()
        shown = 'flat' if score_ is None else f'{score_:.3f}'
        print(f'  {field:24} {shown:>7} {raw or 0:>5.0%} {yours:>5.0f}% '
              f'{theirs:>5.0f}%  {n}')
        report.append({'field': field, 'kappa': score_, 'raw': raw,
                       'human_yes': round(yours, 1), 'judge_yes': round(theirs, 1),
                       'n': n})

    out = human[['model', 'prompt_id', 'replicate']].copy()
    for field in FIELDS:
        out[f'{field}_human'] = human[field]
        out[f'{field}_judge'] = machine[field]
    out.to_csv(ANNOTATION_DIR / 'agreement_rows.csv', index=False)
    pd.DataFrame(report).to_csv(ANNOTATION_DIR / 'agreement.csv', index=False)

    section('Where you differ')
    for field in FIELDS:
        differ = human[human[field] != machine[field]]
        if differ.empty:
            continue
        print(f'\n{field}: {len(differ)} of {len(human)}')
        for row in differ.head(4).itertuples():
            print(f'  {row.prompt_id:24} you {getattr(row, field):<11} '
                  f'judge {machine.loc[row.Index, field]}')
            print(f'    {str(row.response)[:130]}'.replace('\n', ' '))

    print(f'\nWritten to {ANNOTATION_DIR / "agreement.csv"} and '
          f'agreement_rows.csv')


if __name__ == '__main__':
    main()
