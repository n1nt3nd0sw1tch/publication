"""Provide shared pipeline utilities.

Notes
-----
Handle identifiers, files, validation, reporting, and resumable collection loops.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pandas as pd
from settings import (BENCHMARK_COLUMNS, DATASET_NAMES, DATA_DIRS, DOMAIN_NAMES,
                      DRAFTS_COLUMNS, ENV_PATH, CATEGORIES, PROVIDER_KEYS, TYPES)


def make_scenario_id(code, scenario_type, index):
    return f'{code}-{TYPES[scenario_type]["code"]}{index}'


def make_source_id(dataset, record_id):
    return f'{dataset}-{record_id}'


def make_prompt_id(scenario_id, condition):
    return f'{scenario_id}-{condition}'


def code_from_scenario(scenario_id):
    return str(scenario_id).split('-')[0]


def make_prompt(opener, request):
    return f'{opener} {request}'.strip()


def model_slug(model_id):
    return re.sub(r'[^a-z0-9.-]+', '-', str(model_id).lower()).strip('-')


def result_path(model_id, directory):
    return directory / f'{model_slug(model_id)}.jsonl'


def make_directories(directories=DATA_DIRS):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna('')


def read_lines(path):
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


_APPENDING = Lock()


def append_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + '\n'
    with _APPENDING:
        with path.open('a') as file:
            file.write(line)


def read_all(directory):
    frames = [read_lines(path) for path in sorted(directory.glob('*.jsonl'))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def api_key(provider):
    variable = PROVIDER_KEYS.get(provider)
    return environment(variable) if variable else ''


def environment(name):
    if name not in os.environ and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() and not key.strip().startswith('#'):
                os.environ.setdefault(key.strip(), value.strip())
    return os.environ.get(name, '')


_SECTIONS = []


def section(title):
    print(f'\n{title}' if _SECTIONS else title)
    _SECTIONS.append(title)


def shape_of(frame):
    return f'{len(frame)} rows, {frame.shape[1]} columns'


def report(name, problems, notes=()):
    for note in notes:
        print(f'  {name}: {note}')
    if not problems:
        print(f'Validated {name}'
              + (f', {len(notes)} to look at' if notes else ''))
        return
    for problem in problems:
        print(f'  {name}: {problem}')
    raise SystemExit(f'{len(problems)} validation problems in {name}')


def written(scenarios):
    return scenarios[scenarios['request'].str.strip() != ''].reset_index(drop=True)


def validate(frame, required, id_column='', text_columns=(), labels=None):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [f'missing columns {", ".join(missing)}']

    problems = []
    if id_column:
        repeated = frame[id_column][frame[id_column].duplicated()].unique()
        if len(repeated):
            problems.append(f'{len(repeated)} duplicate ids, first {repeated[0]}')
    for column in text_columns:
        blank = frame[column].fillna('').astype(str).str.strip() == ''
        if blank.any():
            problems.append(f'{int(blank.sum())} empty values in {column}')
    for column, allowed in (labels or {}).items():
        values = frame[column].fillna('').astype(str)
        invalid = sorted(set(values) - {str(value) for value in allowed})
        if invalid:
            problems.append(f'invalid {column} labels {", ".join(invalid[:5])}')
    return problems


def check_drafts(drafts):
    problems = validate(frame=drafts, required=DRAFTS_COLUMNS,
                        id_column='source_id',
                        text_columns=['source_id', 'domain'],
                        labels={'domain': DOMAIN_NAMES.values(),
                                'dataset': DATASET_NAMES + [''],
                                'scenario_type': TYPES,
                                'category': CATEGORIES + ['']})
    if problems:
        return problems
    return problems


def check_benchmark(scenarios):
    return validate(frame=scenarios, required=BENCHMARK_COLUMNS,
                    id_column='scenario_id',
                    text_columns=['scenario_id', 'request'],
                    labels={'scenario_type': TYPES,
                            'domain': DOMAIN_NAMES.values(),
                            'dataset': DATASET_NAMES + [''],
                            'category': CATEGORIES})


REPORT_EVERY = 60


WORKERS = 12


def outstanding(wanted, collected, keys):
    if collected.empty:
        return wanted


    def column(name):


        return (collected[name].fillna('') if name in collected.columns
                else pd.Series('', index=collected.index))

    spoiled = ((column('error').astype(str).str.strip() != '')
               | (column('unreadable').astype(str).str.strip() != ''))
    done = {tuple(str(row[key]) for key in keys)
            for (_, row), bad in zip(collected.iterrows(), spoiled) if not bad}
    return [item for item in wanted
            if tuple(str(item[key]) for key in keys) not in done]


def announce(path, wanted, pending, limit=0):
    print(f'{len(wanted) - len(pending)} of {len(wanted)} already collected '
          f'in {path.name}')
    if limit:
        pending = pending[:limit]
    if not pending:
        return []
    print(f'{len(pending)} to collect now')
    return pending


def collect(pending, produce, path, label='', meter=None, produce_batch=None,
            batch_size=1, workers=1, columns=()):
    started, spoke, failures = time.time(), time.time(), 0
    size = batch_size if produce_batch else 1
    index = 0

    def attempt(item):
        try:
            return produce(item), ''
        except Exception as problem:


            return {}, f'{type(problem).__name__}: {problem}'

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(pending), max(size, workers if pool else 1)):
            group = pending[start:start + max(size, workers if pool else 1)]
            if produce_batch:
                try:
                    results = produce_batch(group)
                    errors = [''] * len(group)
                except Exception as problem:
                    results = [{}] * len(group)
                    errors = [f'{type(problem).__name__}: {problem}'] * len(group)
                    failures += len(group)
            elif pool:


                outcomes = list(pool.map(attempt, group))
                results = [result for result, _ in outcomes]
                errors = [error for _, error in outcomes]
                failures += sum(1 for error in errors if error)
            else:
                result, error = attempt(group[0])
                results, errors = [result], [error]
                failures += 1 if error else 0

            for item, result, error in zip(group, results, errors):


                row = {**item, **result, 'error': error}
                append_line(path, {name: row[name] for name in columns
                                   if name in row} if columns else row)
            index += len(group)

            if time.time() - spoke >= REPORT_EVERY or index == len(pending):
                spoke = time.time()
                rate = index / max(time.time() - started, 1)
                line = (f'  {label + "  " if label else ""}{index} of {len(pending)}, '
                        f'{rate * 3600:.0f} an hour, '
                        f'{(len(pending) - index) / rate / 3600:.1f} hours left, '
                        f'{failures} failed')
                if meter is not None:
                    spent, tokens = meter()
                    if spent is not None:
                        projected = spent / index * len(pending)
                        line += (f'\n     ${spent:,.4f} spent, ${projected:,.2f} '
                                 f'projected for this pass, {tokens:,} tokens')
                print(line)
    finally:
        if pool:
            pool.shutdown()
    return failures
