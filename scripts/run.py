"""Run benchmark generation, collection, and provider checks.

Notes
-----
Generate model replies, manage batch workflows, and validate model access.
"""

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import backends
from backends import (BATCH_SIZE, BATCHED, BACKENDS, USAGE, ask, build_payload,
                      call_api, call_ollama,
                      generate, generate_many, provider_of, read_reply,
                      record_usage, spent)
from evaluate import (JUDGE_TEMPERATURE, JUDGE_TOKENS, OLLAMA_JUDGE, build_item,
                      build_policy, compare, describe, read)
from flags import flags_of
from settings import (ADAPTATION_DIR, RESPONSE_COLUMNS, BENCHMARK_PATH, GENERATION, JUDGE,
                      BATCHES_DIR, DIALOGUE_DIR, MODELS, PLAN_PATH, PROMPTS_PATH,
                      PROVIDER_KEYS, TURNS_PATH)
from utils import (WORKERS, announce, api_key, append_line, collect,
                   make_directories, model_slug, outstanding, read_all,
                   read_lines, read_table, result_path, section, shape_of)


TIMEOUT = 30


LISTINGS = {
    'openai': {'url': 'https://api.openai.com/v1/models',
               'headers': lambda key: {'Authorization': f'Bearer {key}'},
               'path': ('data', 'id')},
    'anthropic': {'url': 'https://api.anthropic.com/v1/models',
                  'headers': lambda key: {'x-api-key': key,
                                          'anthropic-version': '2023-06-01'},
                  'path': ('data', 'id')},
    'google': {'url': 'https://generativelanguage.googleapis.com/v1beta/models',
               'headers': lambda key: {'x-goog-api-key': key},
               'path': ('models', 'name')},
}
HUB = 'https://huggingface.co/api/models'


def run_generation(arguments):
    section('Generation')
    prompts = read_table(PROMPTS_PATH)
    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))
    path = result_path(arguments.model, ADAPTATION_DIR)

    wanted = [{'prompt_id': prompt_id, 'model': arguments.model,
               'replicate': replicate, 'backend': arguments.backend,
               'temperature': arguments.temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, arguments.replicates + 1)]
    pending = outstanding(wanted=wanted, collected=read_lines(path),
                          keys=['prompt_id', 'replicate'])

    print(f'{arguments.model} on {arguments.backend}, {len(prompts)} prompts '
          f'times {arguments.replicates} replicates')
    pending = announce(path=path, wanted=wanted, pending=pending,
                       limit=arguments.limit)
    if not pending:
        raise SystemExit('Nothing outstanding')

    def produce(item):


        return {'response': ask(item['backend'], item['model'],
                                by_id[item['prompt_id']], arguments.max_tokens,
                                item['temperature']),
                'blocked': '', 'truncated': False}

    def produce_batch(group):
        replies = generate_many(
            arguments.backend, arguments.model,
            [[{'role': 'user', 'content': by_id[item['prompt_id']]}]
             for item in group],
            arguments.max_tokens, arguments.temperature)
        return [{'response': reply} for reply in replies]

    def meter():
        return (backends.spent(arguments.model),
                backends.USAGE['input'] + backends.USAGE['output'])

    batched = arguments.backend in BATCHED
    if batched:
        print(f'{arguments.batch_size} conversations to the GPU at a time')
    elif arguments.workers > 1:
        print(f'{arguments.workers} calls in flight at once')
    failures = collect(pending=pending, produce=produce, path=path,
                       label=arguments.model, columns=RESPONSE_COLUMNS,
                       meter=meter if arguments.backend == 'api' else None,
                       produce_batch=produce_batch if batched else None,
                       batch_size=arguments.batch_size,
                       workers=arguments.workers)

    section('Collected')
    everything = read_all(ADAPTATION_DIR)
    print(f'{shape_of(everything)} across {ADAPTATION_DIR.name}')
    print(everything.groupby('model').size().to_string())
    spent = backends.spent(arguments.model)
    if spent is not None:
        usage = backends.USAGE
        print(f'\n{usage["calls"]:,} api calls this pass, '
              f'{usage["input"]:,} input and {usage["output"]:,} output tokens, '
              f'${spent:.2f}')
    return failures


def part_path(model, part):
    return BATCHES_DIR / f'part{part}-{model_slug(model)}_output.jsonl'


def joined_path(model):
    return BATCHES_DIR / f'live-{model_slug(model)}_output.jsonl'


def part_items(model, part, parts, replicates=None, temperature=None):
    replicates = GENERATION['replicates'] if replicates is None else replicates
    temperature = GENERATION['temperature'] if temperature is None else temperature
    prompts = read_table(PROMPTS_PATH)
    wanted = [{'prompt_id': prompt_id, 'model': model, 'replicate': replicate,
               'backend': 'api', 'temperature': temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, replicates + 1)]
    pending = outstanding(wanted=wanted, keys=['prompt_id', 'replicate'],
                          collected=read_lines(result_path(model, ADAPTATION_DIR)))
    size = -(-len(wanted) // parts)
    lower, upper = (part - 1) * size, part * size
    order = {(item['prompt_id'], str(item['replicate'])): i
             for i, item in enumerate(wanted)}
    return [item for item in pending
            if lower <= order[(item['prompt_id'], str(item['replicate']))] < upper]


def generate_part(model, part, parts, replicates=None, max_tokens=None,
                  temperature=None, workers=None):
    max_tokens = GENERATION['max_tokens'] if max_tokens is None else max_tokens
    items = part_items(model, part, parts, replicates, temperature)
    path = part_path(model, part)
    if not items:
        return path, 0, 0

    prompts = read_table(PROMPTS_PATH)
    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))


    call = call_ollama if provider_of(model) == 'ollama' else call_api

    def produce(item):
        body = call(model, [{'role': 'user', 'content': by_id[item['prompt_id']]}],
                    max_tokens, item['temperature'])
        append_line(path, {'custom_id': f'{item["prompt_id"]}-r{item["replicate"]}',
                           'response': {'status_code': 200, 'body': body}})
        return {}

    log = path.with_suffix('.log.jsonl')
    failures = collect(pending=items, produce=produce, path=log,
                       label=f'{model} part {part}',
                       meter=lambda: (spent(model),
                                      USAGE['input'] + USAGE['output']),
                       workers=WORKERS if workers is None else workers)


    if failures:
        print(f'  {failures} failed, reasons in {log.name}. They stay '
              f'outstanding, so re-running this part retries them.')
        for reason, count in reasons_in(log).most_common(3):
            print(f'    {count} x {reason[:110]}')
    else:
        log.unlink(missing_ok=True)
    return path, len(items), failures


def reasons_in(log):
    from collections import Counter
    if not log.exists():
        return Counter()
    return Counter(json.loads(line).get('error', '')
                   for line in log.read_text().splitlines()
                   if line.strip() and json.loads(line).get('error'))


def join_parts(model, parts):
    joined = joined_path(model)
    present = [part_path(model, part) for part in range(1, parts + 1)]
    present = [path for path in present if path.exists()]
    if not present:


        already = len(joined.read_text().splitlines()) if joined.exists() else 0
        return joined, already


    def keyed(lines):
        found = {}
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get('custom_id') or record.get('key') or line
            found[key] = line
        return found

    records = keyed(joined.read_text().splitlines() if joined.exists() else [])
    before = len(records)
    for path in present:
        records.update(keyed(path.read_text().splitlines()))

    joined.write_text('\n'.join(records.values()) + '\n')
    for path in present:
        path.unlink()
    added = len(records) - before
    if before and added < len(records):
        print(f'  {before:,} already joined, {added:,} added from '
              f'{len(present)} part file{"s" if len(present) > 1 else ""}')
    return joined, len(records)


def batch_path(model_id, suffix, job_id=""):
    stem = job_id if job_id else f"pending-{model_slug(model_id)}"
    return BATCHES_DIR / f"{stem}_{suffix}.jsonl"


def name_after_job(model_id, job_id):
    pending = batch_path(model_id, "requests")
    named = batch_path(model_id, "requests", job_id)
    if pending.exists():
        pending.replace(named)
    return named


def write_batch(model, endpoint='/v1/responses', replicates=None, max_tokens=None,
                temperature=None, limit=0, fresh=False):
    replicates = GENERATION['replicates'] if replicates is None else replicates
    max_tokens = GENERATION['max_tokens'] if max_tokens is None else max_tokens
    temperature = GENERATION['temperature'] if temperature is None else temperature
    prompts = read_table(PROMPTS_PATH)
    provider = provider_of(model)

    wanted = [{'prompt_id': prompt_id, 'model': model, 'replicate': replicate,
               'backend': 'api', 'temperature': temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, replicates + 1)]


    pending = wanted if fresh else outstanding(
        wanted=wanted, keys=['prompt_id', 'replicate'],
        collected=read_lines(result_path(model, ADAPTATION_DIR)))
    if limit:
        pending = pending[:limit]
    if not pending:
        return None, 0

    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))
    path = batch_path(model, 'requests')
    path.unlink(missing_ok=True)
    for item in pending:
        body = build_payload(
            provider, model,
            [{'role': 'user', 'content': by_id[item['prompt_id']]}],
            max_tokens, item['temperature'])
        custom_id = f'{item["prompt_id"]}-r{item["replicate"]}'


        if provider == 'anthropic':
            line = {'custom_id': custom_id, 'params': body}
        elif provider == 'google':
            line = {'key': custom_id, 'request': body}
        elif provider == 'mistral':


            line = {'custom_id': custom_id,
                    'body': {k: v for k, v in body.items() if k != 'model'}}
        else:
            line = {'custom_id': custom_id, 'method': 'POST', 'url': endpoint,
                    'body': body}
        append_line(path, line)
    return path, len(pending)


def set_aside_replies(model):
    current = result_path(model, ADAPTATION_DIR)
    if not current.exists():
        return None
    aside = ADAPTATION_DIR.parent / 'superseded'
    aside.mkdir(parents=True, exist_ok=True)
    moved = aside / f'{current.stem}.{datetime.now():%Y%m%d-%H%M}.jsonl'
    current.replace(moved)
    return moved


def read_batch(model, source, temperature=None):
    temperature = GENERATION['temperature'] if temperature is None else temperature
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'{source} not found')
    provider = provider_of(model)
    path = result_path(model, ADAPTATION_DIR)

    collected = read_lines(path)
    seen = set() if collected.empty else {
        (str(row.prompt_id), str(row.replicate)) for row in collected.itertuples()}

    read, failed, truncated, repeated, blocked = 0, 0, 0, 0, 0
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)

        prompt_id, _, replicate = str(row.get('custom_id')
                                      or row.get('key')).rpartition('-r')
        if (prompt_id, replicate) in seen:
            repeated += 1
            continue


        if 'key' in row and 'result' not in row:
            body = row.get('response') or {}
            failure = row.get('error') or (None if body else row)
        elif 'result' in row:
            outcome = row['result'] or {}
            body = outcome.get('message') or {}
            failure = None if outcome.get('type') == 'succeeded' else outcome
        else:
            response = row.get('response') or {}
            body = response.get('body') or {}
            failure = (row.get('error')
                       or (body if response.get('status_code', 200) != 200 else None))


        stopped, cut = flags_of(body, row)
        error = ''
        if failure:
            error = str(failure)[:200]
            failed += 1
        elif stopped:
            blocked += 1
        else:
            record_usage(provider, body)


            truncated += cut

        append_line(path, {'prompt_id': prompt_id, 'model': model,
                           'replicate': replicate, 'backend': 'batch',
                           'temperature': temperature, 'error': error,
                           'blocked': stopped, 'truncated': cut,
                           'response': '' if error or stopped
                                       else read_reply(provider, body)})
        seen.add((prompt_id, replicate))
        read += 1
    return read, failed, truncated, repeated, blocked


def run_export(arguments):
    section('Batch export')
    path, count = write_batch(model=arguments.model, endpoint=arguments.endpoint,
                              replicates=arguments.replicates,
                              max_tokens=arguments.max_tokens,
                              temperature=arguments.temperature,
                              limit=arguments.limit, fresh=arguments.fresh)
    if path is None:
        raise SystemExit('Nothing outstanding')
    print(f'{count:,} requests for {arguments.model} on '
          f'{provider_of(arguments.model)}')
    print(f'Written to {path}')
    print(f'\nUpload it in the provider console, endpoint {arguments.endpoint}, '
          f'then when the job finishes:')
    print(f'    python scripts/run.py ingest --model {arguments.model} '
          f'--file <downloaded results>.jsonl')
    return 0


def run_ingest(arguments):
    section('Batch ingest')
    read, failed, truncated, repeated, blocked = read_batch(
        model=arguments.model, source=arguments.file,
        temperature=arguments.temperature)
    print(f'{read:,} replies read into '
          f'{result_path(arguments.model, ADAPTATION_DIR).name}, {failed} failed')
    if repeated:
        print(f'{repeated:,} already collected and skipped')
    if blocked:
        print(f'{blocked} prompts were refused by the provider before reaching '
              f'the model, so they carry no reply')
    if truncated:
        print(f'{truncated} replies hit the {GENERATION["max_tokens"]} token cap, '
              f'so their length is censored rather than measured')
    cost = spent(arguments.model)
    if cost is not None:
        print(f'{USAGE["input"]:,} input and {USAGE["output"]:,} output tokens, '
              f'${cost:,.2f} at the standard rate, ${cost / 2:,.2f} batched')
    return 0


def list_available(provider, key):
    listing = LISTINGS[provider]
    response = requests.get(listing['url'], timeout=TIMEOUT,
                            headers=listing['headers'](key))
    response.raise_for_status()
    collection, field = listing['path']
    return {entry[field].split('/')[-1]
            for entry in response.json().get(collection, [])}


def check_api(spec):
    key = api_key(spec['provider'])
    if not key:
        return f'no {PROVIDER_KEYS[spec["provider"]]} in .env'
    try:
        available = list_available(spec['provider'], key)
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    return 'ok' if spec['id'].split('/')[-1] in available else 'not offered to this key'


def check_local(spec):
    try:
        response = requests.get(f'{HUB}/{spec["id"]}', timeout=TIMEOUT)
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    if response.status_code == 200:
        return 'ok'
    return 'gated, needs a licence accepted' if response.status_code == 401 \
        else f'http {response.status_code}'


def check_panel(models, judge):
    rows = []
    for name, spec in {**models, 'judge': judge}.items():
        verdict = check_api(spec) if spec['access'] == 'api' else check_local(spec)
        rows.append({'model': name, 'provider': spec['provider'],
                     'access': spec['access'], 'weights': spec['weights'],
                     'id': spec['id'], 'status': verdict})
    return pd.DataFrame(rows)


def report_cost(models, generation, prompts):
    calls = prompts * len(models) * generation['replicates']
    section('Run size')
    print(f'Prompts: {prompts}')
    print(f'Models: {len(models)}, replicates: {generation["replicates"]}')
    print(f'Replies to generate: {calls}')
    print(f'Replies to judge: {calls}')


def load_prompt(prompt_id, prompts_path, benchmark_path):
    prompts = read_table(prompts_path)
    benchmark = read_table(benchmark_path)
    chosen = prompts[prompts['prompt_id'] == prompt_id] if prompt_id else prompts
    if chosen.empty:
        raise KeyError(f'{prompt_id} is not in {prompts_path.name}')
    row = chosen.iloc[0]
    return row, benchmark.set_index('scenario_id').loc[row['scenario_id']]


def show(title, text, width=76):
    print(f'{title}:')
    for line in textwrap.wrap(str(text), width=width) or ['']:
        print(f'  {line}')


def trace(arguments):
    row, scenario = load_prompt(prompt_id=arguments.prompt_id,
                                prompts_path=PROMPTS_PATH,
                                benchmark_path=BENCHMARK_PATH)

    section('Prompt')
    print(f'{row["prompt_id"]}  {scenario["domain"]} / '
          f'{scenario["scenario_type"]} / {row["condition"]}')
    show('Sent to the model', row['prompt'])

    section('Reply')
    if arguments.reply:
        print('Supplied, not generated')
        reply = arguments.reply
    else:
        print(f'Generating with {arguments.model} on {arguments.backend}')
        reply = ask(arguments.backend, arguments.model, row['prompt'])
    show('Returned', reply)

    section('Judgement')
    if arguments.verdict:
        print('Supplied, classifier not called')
        output = arguments.verdict
    else:
        print(f'Scoring with {arguments.judge}')


        output = generate(arguments.backend, arguments.judge,
                          [{'role': 'system', 'content': build_policy()},
                           {'role': 'user',
                            'content': build_item(scenario['request'], reply)}],
                          max_tokens=JUDGE_TOKENS,
                          temperature=JUDGE_TEMPERATURE)
    verdict, problems = read(output)
    for problem in problems:
        print(f'  Unreadable, {problem}')
    for field, value in (verdict or {}).items():
        print(f'  {field:<20} {value}')

    section('Comparison')
    expected = row['expected_answer']
    observed = (verdict or {}).get('answer', '')
    deviation = compare(observed=observed, expected=expected)
    print(f'  Expected             {expected or "None, this is the control"}')
    print(f'  Observed             {observed or "not read"}')
    print(f'  Deviation            '
          f'{deviation if deviation is not None else "n/a"}  ({describe(deviation)})')


def run_check(arguments):
    section('Model panel')
    panel = check_panel(models=MODELS, judge=JUDGE)
    print(f'Panel: {shape_of(panel)}')
    print(panel.to_string(index=False))
    failed = panel[panel['status'] != 'ok']
    if len(failed):
        print(f'\n{len(failed)} models are not reachable')

    prompts = read_table(PROMPTS_PATH) if PROMPTS_PATH.exists() else pd.DataFrame()
    report_cost(models=MODELS, generation=GENERATION, prompts=len(prompts))


    if arguments.model or arguments.reply:
        trace(arguments)
    return 0


def dialogue_path(model):
    return DIALOGUE_DIR / f'{model_slug(model)}.jsonl'


def history_before(turns, turn):
    return [{'role': row['role'], 'content': row['text']}
            for _, row in turns.iterrows()
            if int(row['turn']) < int(turn)]


def dialogue_items(model):
    plan = read_table(PLAN_PATH)
    plan = plan[plan['model'] == model]
    wanted = [{'dialogue_id': name, 'model': model}
              for name in plan['dialogue_id'].unique()]


    have = read_lines(dialogue_path(model))
    if have.empty:
        return wanted
    per_dialogue = have.groupby('dialogue_id').size()
    needed = (plan[(plan['role'] == 'assistant')
                   & (plan['turn'].astype(int) > 2)]
              .groupby('dialogue_id').size())
    done = {name for name, count in per_dialogue.items()
            if count >= needed.get(name, 0)}
    return [item for item in wanted if item['dialogue_id'] not in done]


def collect_dialogues(model, backend='api', max_tokens=None, temperature=None,
                      workers=1):
    pending = dialogue_items(model)
    path = dialogue_path(model)
    if not pending:
        return path, 0, 0

    plan = read_table(PLAN_PATH)
    plan = plan[plan['model'] == model]
    plan = plan.assign(turn=plan['turn'].astype(int)).sort_values('turn')
    grouped = {name: rows for name, rows in plan.groupby('dialogue_id')}

    order = {name: (rows.iloc[0]['prompt_id'], name)
             for name, rows in grouped.items()}
    pending.sort(key=lambda item: order[item['dialogue_id']])


    have = read_lines(path)
    existing = ({} if have.empty else
                {(row['dialogue_id'], str(row['turn'])): row['text']
                 for _, row in have.iterrows()})

    def produce(item):


        messages = []
        for _, row in grouped[item['dialogue_id']].iterrows():
            if row['role'] == 'user':
                messages.append({'role': 'user', 'content': row['text']})
                continue


            text = '' if pd.isna(row['text']) else str(row['text']).strip()
            if not text:
                key = (item['dialogue_id'], str(row['turn']))
                if key in existing:
                    text = existing[key]
                else:
                    text = generate(backend, model, messages,
                                    max_tokens, temperature)
                    append_line(path, {'dialogue_id': item['dialogue_id'],
                                       'turn': row['turn'], 'model': model,
                                       'text': text})
            messages.append({'role': 'assistant', 'content': text})
        return {}

    log = path.with_suffix('.log.jsonl')
    failures = collect(pending=pending, produce=produce, path=log,
                       label=f'{model} dialogues',
                       meter=lambda: (spent(model),
                                      USAGE['input'] + USAGE['output']),
                       workers=workers)
    if failures:
        print(f'  {failures} failed, reasons in {log.name}. They stay '
              f'outstanding, so running again retries them.')
        for reason, count in reasons_in(log).most_common(3):
            print(f'    {count} x {reason[:110]}')
    else:
        log.unlink(missing_ok=True)
    return path, len(pending), failures


def run_dialogue(arguments):
    if not PLAN_PATH.exists():
        raise SystemExit('No plan.csv, run: python scripts/build.py turns')

    model = arguments.model
    section(f'Dialogues, {model}')
    backend = arguments.backend if provider_of(model) == 'ollama' else 'api'
    path, asked, failures = collect_dialogues(
        model=model, backend=backend,
        max_tokens=arguments.max_tokens, temperature=arguments.temperature,
        workers=arguments.workers)
    print(f'{asked:,} dialogues asked for, {failures:,} failed, '
          f'written to {path.name}')
    return failures


def merge_turns():
    plan = read_table(PLAN_PATH)
    collected = {}
    for model in plan['model'].unique():


        frame = read_lines(dialogue_path(model))
        if frame.empty:
            continue
        collected.update({(row['dialogue_id'], str(row['turn'])): row['text']
                          for _, row in frame.iterrows()})

    plan['text'] = [collected.get((row['dialogue_id'], str(row['turn'])),
                                  row['text'])
                    for _, row in plan.iterrows()]

    generated = plan[(plan['role'] == 'assistant')
                     & (plan['turn'].astype(int) > 2)]
    empty = generated[generated['text'].astype(str).str.strip() == '']
    incomplete = set(empty['dialogue_id'])
    if incomplete:
        print(f'{len(incomplete):,} dialogues are missing a generated turn '
              f'and are dropped')
        for model, count in (empty.drop_duplicates('dialogue_id')['model']
                             .value_counts().items()):
            print(f'   {model}: {count}')
        plan = plan[~plan['dialogue_id'].isin(incomplete)]

    TURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(TURNS_PATH, index=False)
    print(f'{plan["dialogue_id"].nunique():,} complete dialogues, '
          f'{len(plan):,} turns, written to {TURNS_PATH.name}')
    return plan


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',
                        choices=['check', 'generate', 'dialogue', 'export', 'ingest'])
    parser.add_argument('--model', default='')
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--judge', default='')
    parser.add_argument('--replicates', type=int, default=GENERATION['replicates'])
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, to time a pass')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='conversations handed to vLLM at once')
    parser.add_argument('--workers', type=int, default=1,
                        help='api calls in flight at once, for a provider with '
                             'no batch queue')
    parser.add_argument('--endpoint', default='/v1/responses',
                        help='the endpoint a batch job runs against')
    parser.add_argument('--file', default='',
                        help='the results file downloaded from the console')
    parser.add_argument('--force', action='store_true',
                        help='export the whole pass again, ignoring what is '
                             'already collected')
    parser.add_argument('--fresh', action='store_true',
                        help='export every prompt again, ignoring what is collected')
    parser.add_argument('--prompt-id', default='', help='the prompt to trace')
    parser.add_argument('--reply', default='',
                        help='skip generation and score this text instead')
    parser.add_argument('--verdict', default='',
                        help='skip the classifier and read this output instead')
    return parser


if __name__ == '__main__':
    arguments = parser().parse_args()
    if not arguments.judge:
        arguments.judge = JUDGE['id']

    make_directories()
    if arguments.stage in ('generate', 'dialogue', 'export', 'ingest') \
            and not arguments.model:
        raise SystemExit(f'--model is needed to {arguments.stage}')
    if arguments.stage == 'generate':
        failures = run_generation(arguments)
    elif arguments.stage == 'dialogue':
        failures = run_dialogue(arguments)
    elif arguments.stage == 'export':
        failures = run_export(arguments)
    elif arguments.stage == 'ingest':
        failures = run_ingest(arguments)
    else:
        failures = run_check(arguments)

    if failures:
        print(f'\n{failures} failed this pass, run again to retry them')
