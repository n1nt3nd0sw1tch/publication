"""Build the benchmark prompts and multi-turn dialogue plan.

Notes
-----
The script constructs benchmark prompts from configured source corpora,
validates the experimental design, and prepares the dialogue extension.
"""

import argparse
import itertools
import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
from settings import (ADAPTATION_DIR, ANSWERS, BANDS, BENCHMARK_COLUMNS, CATEGORIES,
                      BENCHMARK_PATH, CONDITION_AGES, CONDITION_NAMES,
                      CONDITIONS, CUES, PLAN_PATH, DIALOGUE_COLUMNS,
                      DOMAIN_CODES, DOMAIN_NAMES, DRAFTS_COLUMNS, DRAFTS_PATH,
                      DRAFTS_WRITTEN,
                      METHODS, AUTHORED, MINOR, MINOR_BANDS, ORIGINAL_DIR, DIALOGUE, PROMPTS_PATH, WITHHELD_PATH, PROMPT_COLUMNS, SCENARIOS,
                      SCENARIOS_PATH, SEED, SIGNALS, SOURCES, TOTAL_SCENARIOS,
                      TYPES, TYPE_ANSWERS)
from utils import (check_benchmark, check_drafts, code_from_scenario,
                   make_prompt, make_prompt_id, make_scenario_id,
                   make_source_id, read_all, read_table, report, section,
                   validate, written)


KEYWORD_RULES = {
    'bod': (r'\b(?:body image|look fat|too fat|fat people|skinny girls|'
            r'how much should i weigh|ugly)\b'),
    'eat': (r'\b(?:anorex|bulimi|purg|starv|throw up|vomit|thinspo|fasting|'
            r'calorie restriction)\w*'),
    'vio': r'\b(?:assassinat\w*|john f kennedy|shinzo abe)',
}

EXPECTED_PROMPTS = sum(TOTAL_SCENARIOS for _ in CONDITIONS)


MIN_WORDS, MAX_WORDS = 5, 14


NORMAL_CELL = {'Harmful': 5, 'Age Restricted': 5, 'Rights': 5, 'Benign': 5}
WITHOUT_RESTRICTED = {'Harmful': 5, 'Rights': 10, 'Benign': 5}

CATEGORY_OPENINGS = {


    'Instruction': r'^(write|tell|explain|give|teach|describe|show|help|make|'
                   r'point|promise)\b|^can you\b|^how (do|can|would) i\b',


    'Information': r'^(what|why|where|when|which|who|whom)\b|'
                   r'^how (many|much|long|often)\b|^(can|may|am) i\b|'
                   r'^(is|are|was|were|do|does|did|should|would|could) \w',
}


CONSTRAINING = (r'\b(short|shorter|brief|briefly|concise|long|lengthy|detailed|'
                r'detail|list|bullet|bullets|paragraph|paragraphs|sentence|'
                r'sentences|words?|essay|summary|summarise|outline|table)\b')


SIMILARITY_CEILING = 0.70


AGE_MARKED = (r'\b(school|teacher|mum|mummy|dad|daddy|parent|parents|'
              r'grown.?up|grown.?ups|kid|kids|child|children|homework|'
              r'classmate|classmates|colleague|colleagues|boss|nursery|'
              r'toddler|pupil|pension|retired)\b')


def check_scenarios(scenarios, types=TYPES):
    problems = []
    for domain, given in scenarios.items():


        restricted = len(given.get('Age Restricted', []))
        expected = dict(NORMAL_CELL) if restricted else dict(WITHOUT_RESTRICTED)
        for scenario_type, wanted in expected.items():
            count = len(given.get(scenario_type, []))
            if count != wanted:
                problems.append(f'{domain} has {count} {scenario_type} '
                                f'scenarios, expected {wanted}')
        for scenario_type in given:
            if scenario_type not in expected:
                problems.append(f'{domain} has a {scenario_type} stratum, '
                                f'which this shape does not allow')
    entries = [(domain, scenario_type, entry)
               for domain, given in scenarios.items()
               for scenario_type, values in given.items() for entry in values]
    return (problems + check_fields(entries) + check_length(entries)
            + check_categories(entries, types) + check_distinct(entries)
            + check_neutral(entries) + check_unconstrained(entries))


def check_fields(entries):
    problems = []
    for domain, scenario_type, entry in entries:
        missing = {'source', 'category', 'base'} - set(entry)
        if missing:
            problems.append(f'{domain} {scenario_type} '
                            f'{entry.get("base", "?")!r} is missing '
                            f'{", ".join(sorted(missing))}')
            continue
        category, base = entry['category'], entry['base']
        if category not in CATEGORIES:
            problems.append(f'{base!r} has category {category!r}')
        elif not re.match(CATEGORY_OPENINGS[category], base, re.I):
            problems.append(f'{base!r} is written down as {category} but does not '
                            f'open as one')
    return problems


def check_unconstrained(entries, pattern=CONSTRAINING):
    return [f'{entry["base"]!r} contains {match.group(0)!r}, which constrains '
            f'the reply' for _, _, entry in entries
            if (match := re.search(pattern, entry['base'], re.I))]


def check_length(entries, low=MIN_WORDS, high=MAX_WORDS):
    return [f'{entry["base"]!r} is {len(entry["base"].split())} words'
            for _, _, entry in entries
            if not low <= len(entry['base'].split()) <= high]


def check_categories(entries, types):
    problems = []
    cells = {}
    for domain, scenario_type, entry in entries:
        cells.setdefault((domain, scenario_type), []).append(entry['category'])


    for (domain, scenario_type), forms in sorted(cells.items()):
        counts = Counter(forms)
        if scenario_type == 'Age Restricted':
            if counts.get('Information', 0):
                problems.append(f'{domain} {scenario_type} has '
                                f'{counts["Information"]} Information requests, '
                                f'and this stratum is procedural throughout')
            continue
        missing = [category for category in CATEGORIES if not counts.get(category)]
        if missing:
            problems.append(f'{domain} {scenario_type} splits {dict(counts)}, '
                            f'with no {" or ".join(missing)} request at all')
    return problems


def check_distinct(entries, ceiling=SIMILARITY_CEILING):
    problems = []
    for first, second in itertools.combinations(entries, 2):
        ratio = SequenceMatcher(None, first[2]['base'].lower(),
                                second[2]['base'].lower()).ratio()
        if ratio < ceiling:
            continue


        if first[1] == second[1] == 'Age Restricted':
            continue
        problems.append(f'{ratio:.2f} similar: {first[2]["base"]!r} and '
                        f'{second[2]["base"]!r}')
    return problems


def check_neutral(entries, pattern=AGE_MARKED):
    found = [(entry['base'], match.group(0)) for _, _, entry in entries
             if (match := re.search(pattern, entry['base'], re.I))]
    return [f'{base!r} contains {word!r}, which fixes an age' for base, word in found]


def fill(drafts, scenarios):
    written_count, added = 0, []
    for domain, types in scenarios.items():
        number = 0
        for scenario_type, entries in types.items():
            for entry in entries:
                source_id = '' if entry['source'] == 'authored' else entry['source']
                base = entry['base']
                number += 1
                values = {'domain': domain, 'scenario_type': scenario_type,
                          'category': entry['category'],


                          'order': number,


                          'request': f'{base}.' if entry['category'] == 'Instruction'
                          else f'{base}?'}
                rows = (drafts.index[drafts['source_id'] == source_id]
                        if source_id else [])
                if len(rows):
                    for column, value in values.items():
                        drafts.at[rows[0], column] = value
                    written_count += 1
                else:
                    code = DOMAIN_CODES[domain]
                    added.append({'source_id': f'authored-{code}-{number}',
                                  'dataset': AUTHORED, 'source_prompt': '',
                                  **values})
    filled = pd.concat([drafts, pd.DataFrame(added)], ignore_index=True)
    return filled[DRAFTS_COLUMNS], written_count, len(added)


def load_original(filename, original_dir):
    path = original_dir / filename
    if not path.exists():
        print(f'Skipped {filename}, not downloaded')
        return None
    return pd.read_csv(path)


def read_labels(frame, label, split=''):
    labels = frame[label].astype(str)
    return labels.str.split(split).str[0].str.strip() if split else labels.str.strip()


def map_domains(frame, label, domains, split=''):
    lowered = {str(value).lower(): code for code, values in domains.items()
               for value in values}
    return read_labels(frame, label, split).str.lower().map(lowered)


def select(name, spec, original_dir):
    frame = load_original(filename=spec['file'], original_dir=original_dir)
    if frame is None:
        return None
    for column, allowed in spec.get('keep', {}).items():
        frame = frame[frame[column].isin(allowed)]
    if spec.get('exclude'):
        frame = frame[~frame[spec['record']].isin(spec['exclude'])]

    needed = [spec['text'], spec['label'], *spec.get('keep', {}),
              *([spec['record']] if spec['record'] else []),
              *([spec['fallback']['label']] if spec.get('fallback') else [])]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f'{spec["file"]} is missing columns {", ".join(missing)}')

    codes = map_domains(frame=frame, label=spec['label'], domains=spec['domains'],
                        split=spec.get('split', ''))
    if spec.get('fallback'):
        codes = codes.fillna(map_domains(frame=frame, **spec['fallback']))

    records = frame[spec['record']] if spec['record'] else frame.index
    selected = pd.DataFrame({
        'dataset': name,
        'record_id': list(records),
        'source_prompt': frame[spec['text']].astype(str).str.strip().tolist(),
        'domain_code': codes.tolist(),
    })
    return selected[selected['domain_code'].notna()].reset_index(drop=True)


def apply_keyword_rules(sources, rules):
    assigned = pd.Series(False, index=sources.index)
    for code, pattern in rules.items():
        matches = sources['source_prompt'].str.contains(
            pattern, case=False, regex=True) & ~assigned
        sources.loc[matches, 'domain_code'] = code
        assigned = assigned | matches
    return sources, int(assigned.sum())


def remove_duplicates(sources):
    normalised = (sources['source_prompt'].str.lower()
                  .str.replace(r'[^a-z0-9\s]', '', regex=True)
                  .str.replace(r'\s+', ' ', regex=True).str.strip())
    repeated = sources.assign(normalised=normalised) \
        .duplicated(subset=['domain_code', 'normalised'])
    return sources.loc[~repeated].reset_index(drop=True), int(repeated.sum())


def assign_ids(sources):
    sources = sources.assign(
        source_id=[make_source_id(dataset, record) for dataset, record
                   in zip(sources['dataset'], sources['record_id'])],
        domain=sources['domain_code'].map(DOMAIN_NAMES))
    return sources.sort_values(['domain', 'source_id']).reset_index(drop=True)


def build_sources(sources, original_dir):
    frames = [select(name=name, spec=spec, original_dir=original_dir)
              for name, spec in sources.items()]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError('No raw data found, run download.py first')

    records = pd.concat(frames, ignore_index=True)
    records, moved = apply_keyword_rules(sources=records, rules=KEYWORD_RULES)
    records, repeated = remove_duplicates(sources=records)
    records = assign_ids(sources=records)
    print(f'{len(records)} usable records from '
          f'{records["dataset"].nunique()} datasets, {moved} reassigned by '
          f'wording, {repeated} duplicates removed')
    return records


def build_drafts(sources):
    return pd.DataFrame({
        'source_id': sources['source_id'],
        'dataset': sources['dataset'].map(
            {name: spec['name'] for name, spec in SOURCES.items()}),
        'domain': sources['domain'],
        'scenario_type': sources['dataset'].map(
            {name: spec['scenario_type'] for name, spec in SOURCES.items()}),
        'category': '',
        'order': 0,
        'source_prompt': sources['source_prompt'],
        'request': '',
    })[DRAFTS_COLUMNS]


def build_benchmark(drafts, domains, types):
    kept = drafts[drafts['request'].str.strip() != '']
    rows = []
    for code, name in domains.items():
        for scenario_type, values in types.items():
            chosen = kept[(kept['domain'] == name)
                          & (kept['scenario_type'] == scenario_type)] \
                .sort_values('order')


            for index in range(1, max(len(chosen), 1) + 1):
                draft = chosen.iloc[index - 1] if index <= len(chosen) else None
                if draft is None and scenario_type == 'Age Restricted':
                    continue
                rows.append({
                    'scenario_id': make_scenario_id(code, scenario_type, index),
                    'dataset': draft['dataset'] if draft is not None else '',
                    'domain': name,
                    'scenario_type': scenario_type,


                    **{column: draft[column] if draft is not None else ''
                       for column in ['category', 'source_id', 'request']},
                })
    return pd.DataFrame(rows)[BENCHMARK_COLUMNS]


def expected_answer(scenario_type, band):
    answers = TYPE_ANSWERS[scenario_type]
    if not answers:
        return ''
    if band == MINOR:
        given = {answers[b] for b in MINOR_BANDS}
        return given.pop() if len(given) == 1 else ''
    if band:
        return answers[band]
    given = set(answers.values())
    return given.pop() if len(given) == 1 else ''


def build_prompt(scenario, condition):
    request = scenario['request'].strip()
    if not request:
        return None
    return {
        'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name']),
        'scenario_id': scenario['scenario_id'],
        'condition': condition['name'],
        'age': condition['age'],
        'band': condition['band'],
        'signal': condition['signal'],
        'cue': condition['cue'],
        'opener': condition['opener'],
        'request': request,
        'prompt': make_prompt(condition['opener'], request),
        'expected_answer': expected_answer(scenario['scenario_type'],
                                           condition['band']),
    }


def build_prompts(scenarios, conditions):
    rows = [build_prompt(scenario=scenario, condition=condition)
            for _, scenario in scenarios.iterrows() for condition in conditions]
    return pd.DataFrame([row for row in rows if row])[PROMPT_COLUMNS]


def check_identity(prompts):
    wrong = [row.prompt_id for row in prompts.itertuples()
             if row.prompt != make_prompt(row.opener, row.request)]
    if wrong:
        return [f'{len(wrong)} prompts are not their opener followed by their '
                f'request, first {wrong[0]}']
    varying = prompts.groupby('scenario_id')['request'].nunique()
    drifted = varying[varying > 1]
    if len(drifted):
        return [f'{len(drifted)} scenarios carry more than one request across '
                f'their conditions, first {drifted.index[0]}']
    return []


def check_prompts(prompts):
    problems = validate(frame=prompts, required=PROMPT_COLUMNS,
                        id_column='prompt_id',
                        text_columns=['prompt_id', 'scenario_id', 'prompt',
                                      'request'],
                        labels={'condition': CONDITION_NAMES, 'signal': SIGNALS,
                                'cue': CUES, 'band': BANDS + [''],
                                'age': CONDITION_AGES,
                                'expected_answer': ANSWERS + ['']})
    return problems + check_identity(prompts)


def report_prompts(prompts):
    counts = prompts.groupby('signal').size()
    signals = ', '.join(f'{counts.get(name, 0)} {name.lower()}'
                        for name in ['Explicit', 'Implicit', 'None'])
    print(f'{len(prompts)} prompts, {prompts["scenario_id"].nunique()} scenarios '
          f'by {len(CONDITIONS)} conditions')
    print(f'{signals.replace("none", "without a signal")}')


RESPONSE_COLUMNS = ['prompt_id', 'model', 'replicate', 'response', 'error']


def load_responses(directory=ADAPTATION_DIR):
    responses = read_all(directory)
    if responses.empty:
        raise FileNotFoundError(
            f'Nothing in {directory.name}, run the generate stage first')
    missing = [column for column in RESPONSE_COLUMNS
               if column not in responses.columns]
    if missing:
        raise KeyError(f'{directory.name} is missing columns '
                       f'{", ".join(missing)}')

    responses = responses[responses['error'].astype(str).str.strip() == '']


    empty = responses['response'].astype(str).str.strip() == ''
    if empty.any():
        withheld = responses[empty]
        print(f'{len(withheld)} openings carry no reply and cannot be replayed')
        for model, count in withheld['model'].value_counts().items():
            print(f'   {model}: {count}')
        WITHHELD_PATH.parent.mkdir(parents=True, exist_ok=True)
        withheld.to_csv(WITHHELD_PATH, index=False)
        responses = responses[~empty]

    return responses.astype({'replicate': str})


def choose_scenarios(prompts, count, seed, strata=None):
    """Draw the dialogue subset, whole strata first and the remainder
    stratified across harm domains.

    strata restricts the pool to the scenario types that refuse often enough to
    leave something to measure. Any stratum small enough to fit inside the count
    is taken entire rather than sampled, which is what keeps all 25
    age-restricted scenarios in: they are the only type whose expected answer
    moves with age, so sampling them would weaken the contrast the extension
    exists to test. The remainder is drawn from the larger strata under the
    seed, balanced across domains.
    """
    scenarios = prompts[['scenario_id']].drop_duplicates()
    scenarios['domain'] = scenarios['scenario_id'].map(code_from_scenario)
    scenarios['scenario_type'] = scenarios['scenario_id'].map(
        lambda name: next(kind for kind, values in TYPES.items()
                          if values['code'] == name.split('-')[1][0]))

    if strata:
        scenarios = scenarios[scenarios['scenario_type'].isin(strata)]
        if scenarios.empty:
            raise ValueError(f'No scenarios in strata {", ".join(strata)}')


    order = scenarios['scenario_type'].value_counts(ascending=True).index
    chosen, remaining = [], count

    for index, stratum in enumerate(order):
        pool = scenarios[scenarios['scenario_type'] == stratum]
        share = remaining if index == len(order) - 1 else round(
            remaining / (len(order) - index))
        take = min(len(pool), share)

        if take >= len(pool):
            chosen.extend(pool['scenario_id'])
        else:


            sizes = pool.groupby('domain').size()
            exact = sizes * take / len(pool)
            quota = exact.astype(int)
            short = take - int(quota.sum())
            if short > 0:

                order_by_remainder = sorted(
                    sizes.index,
                    key=lambda name: (-(exact[name] - quota[name]), name))
                for name in order_by_remainder[:short]:
                    quota[name] += 1
            quota = quota.clip(upper=sizes)

            drawn = [group.sample(n=int(quota[name]), random_state=seed)
                     for name, group in pool.groupby('domain')
                     if quota[name] > 0]
            chosen.extend(sorted(pd.concat(drawn)['scenario_id'])[:take])

        remaining -= take
        if remaining <= 0:
            break

    if len(chosen) < count:
        print(f'Only {len(chosen)} scenarios available, {count} requested')

    return sorted(chosen)


def build_dialogue(prompt, reply, method, turns, request):
    slug = method.lower().replace(' ', '-')
    dialogue_id = (f'{prompt["prompt_id"]}-{reply["model"]}'
                   f'-r{reply["replicate"]}-{slug}')
    shared = {'dialogue_id': dialogue_id, 'prompt_id': prompt['prompt_id'],
              'scenario_id': prompt['scenario_id'],
              'condition': prompt['condition'], 'band': prompt['band'],
              'model': reply['model'], 'opening_replicate': reply['replicate'],
              'method': method}
    rows = [{**shared, 'turn': 1, 'role': 'user', 'text': prompt['prompt'],
             'expected_answer': prompt['expected_answer']},
            {**shared, 'turn': 2, 'role': 'assistant', 'text': reply['response'],
             'expected_answer': ''}]
    for index, wording in enumerate(turns):
        turn = 3 + index * 2
        rows.append({**shared, 'turn': turn, 'role': 'user',
                     'text': wording.format(request=request),
                     'expected_answer': ''})
        rows.append({**shared, 'turn': turn + 1, 'role': 'assistant', 'text': '',
                     'expected_answer': prompt['expected_answer']})
    return rows


def build_dialogues(prompts, responses, requests, methods, scenarios,
                    conditions, opening_replicate):
    wanted = prompts[prompts['scenario_id'].isin(scenarios)
                     & prompts['condition'].isin(conditions)]


    setting = str(opening_replicate).lower()
    if setting == 'all':
        opening = responses
    elif setting == 'first':
        opening = responses[responses['replicate'] == '1']
    else:
        opening = responses[responses['replicate'] == str(opening_replicate)]

    if opening.empty:
        raise ValueError(
            f'opening_replicate {opening_replicate!r} matched no replies. '
            f'The replicate column holds '
            f'{", ".join(sorted(responses["replicate"].unique()))}.')
    merged = wanted.merge(opening, on='prompt_id', how='inner')
    rows = [row for _, pair in merged.iterrows() for method in methods
            for row in build_dialogue(prompt=pair, reply=pair, method=method,
                                      turns=METHODS[method]['turns'],
                                      request=requests[pair['scenario_id']])]
    if not rows:
        raise ValueError(
            f'No dialogues to build. {len(wanted)} prompts match the chosen '
            f'scenarios and conditions and {len(opening)} openings match the '
            f'replicate, but nothing joins on prompt_id.')
    return pd.DataFrame(rows)[DIALOGUE_COLUMNS]


def check_dialogues(dialogues, methods):
    problems = validate(frame=dialogues, required=DIALOGUE_COLUMNS,
                        text_columns=['dialogue_id', 'prompt_id', 'scenario_id'],
                        labels={'role': ['user', 'assistant'],
                                'band': BANDS + [''],
                                'method': list(METHODS),
                                'expected_answer': ANSWERS + ['']})
    turns = 2 + 2 * len(METHODS[methods[0]]['turns'])
    counts = dialogues.groupby('dialogue_id').size()
    uneven = counts[counts != turns]
    if len(uneven):
        problems.append(f'{len(uneven)} dialogues do not have {turns} turns')


    numbered = pd.to_numeric(dialogues['turn'], errors='coerce')
    replayed = dialogues[(numbered == 2)
                         & (dialogues['text'].astype(str).str.strip() == '')]
    if len(replayed):
        problems.append(f'{len(replayed)} dialogues have an empty replayed reply')

    generated = dialogues[(numbered > 2) & (dialogues['role'] == 'assistant')
                          & (dialogues['text'].astype(str).str.strip() != '')]
    if len(generated):
        problems.append(f'{len(generated)} later assistant turns are already '
                        f'filled, which should happen at generation time')
    return problems


def report_dialogues(dialogues, methods, scenarios):
    numbered = pd.to_numeric(dialogues['turn'], errors='coerce')
    generated = int(((dialogues['role'] == 'assistant') & (numbered > 2)).sum())
    print(f'{dialogues["dialogue_id"].nunique()} conversations from '
          f'{len(scenarios)} scenarios, '
          f'{2 + 2 * len(METHODS[methods[0]]["turns"])} turns each, '
          f'{generated} replies to generate')
    opening = dialogues[numbered == 1]
    print(pd.crosstab(opening['condition'], opening['method'],
                      margins=True, margins_name='total').to_string())


def build_all():
    section('Source records')
    records = build_sources(sources=SOURCES, original_dir=ORIGINAL_DIR)

    section('Scenarios')
    found = check_scenarios(SCENARIOS)
    advisory = ('words' in problem or 'similar:' in problem
                for problem in found)
    report(SCENARIOS_PATH.name,
           [p for p, note in zip(found, list(advisory)) if not note],
           notes=[p for p in found
                  if 'words' in p or 'similar:' in p])
    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    drafts = build_drafts(sources=records)


    drafts, adapted, authored = fill(drafts=drafts, scenarios=SCENARIOS)
    print(f'{adapted + authored} written, {adapted} adapted from a source '
          f'record and {authored} authored')

    section('Drafts')
    drafts[DRAFTS_WRITTEN].to_csv(DRAFTS_PATH, index=False)
    report('drafts.csv', check_drafts(drafts))
    print(f'{len(drafts)} drafts, {int((drafts["request"].str.strip() != "").sum())} '
          f'carrying a request')

    section('Benchmark')
    benchmark = build_benchmark(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)
    benchmark.to_csv(BENCHMARK_PATH, index=False)
    filled = written(benchmark)
    if filled.empty:
        raise SystemExit('No scenarios written yet, nothing further to build.')
    report('benchmark.csv', check_benchmark(filled))
    print(f'{len(filled)} scenarios across {filled["domain"].nunique()} categories '
          f'and {filled["scenario_type"].nunique()} types')

    section('Prompts')
    prompts = build_prompts(scenarios=filled, conditions=CONDITIONS)
    report('prompts.csv', check_prompts(prompts))
    prompts.to_csv(PROMPTS_PATH, index=False)
    report_prompts(prompts=prompts)


def build_turns():
    section('Dialogue extension')
    prompts = read_table(PROMPTS_PATH)
    responses = load_responses()
    benchmark = read_table(BENCHMARK_PATH)


    requests = dict(zip(benchmark['scenario_id'], benchmark['request']))

    scenarios = choose_scenarios(prompts=prompts,
                                 count=DIALOGUE['scenarios'], seed=SEED,
                                 strata=DIALOGUE.get('strata'))
    methods = DIALOGUE['methods']
    dialogues = build_dialogues(prompts=prompts, responses=responses,
                                requests=requests, methods=methods,
                                scenarios=scenarios,
                                conditions=DIALOGUE['conditions'],
                                opening_replicate=DIALOGUE['opening_replicate'])
    report('plan.csv', check_dialogues(dialogues=dialogues, methods=methods))

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    dialogues.to_csv(PLAN_PATH, index=False)
    report_dialogues(dialogues=dialogues, methods=methods, scenarios=scenarios)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', nargs='?', default='benchmark',
                        choices=['benchmark', 'turns'])
    arguments = parser.parse_args()
    build_all() if arguments.stage == 'benchmark' else build_turns()
