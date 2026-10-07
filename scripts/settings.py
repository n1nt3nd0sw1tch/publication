"""Expose study configuration, paths, and derived design constants.

Notes
-----
Load validated YAML settings and provide shared paths and schema definitions.
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
SETTINGS_PATH = CONFIG_DIR / 'settings.yml'
SCENARIOS_PATH = CONFIG_DIR / 'scenarios.yml'
JUDGE_PATH = CONFIG_DIR / 'judge.yml'
ENV_PATH = ROOT / '.env'


DATA_DIR = ROOT / 'data'


BENCHMARK_PATH = DATA_DIR / 'benchmark.csv'
PROMPTS_PATH = DATA_DIR / 'prompts.csv'


SOURCES_PATH = DATA_DIR / 'sources.md'


PROCESS_DIR = DATA_DIR / 'process'
ORIGINAL_DIR = PROCESS_DIR / 'corpora'
DRAFTS_PATH = PROCESS_DIR / 'drafts.csv'
AOA_PATH = ORIGINAL_DIR / 'aoa.csv'


BATCHES_DIR = DATA_DIR / 'batches'


RESULTS_DIR = ROOT / 'results'
ADAPTATION_DIR = RESULTS_DIR / 'adaptation'

DIALOGUE_DIR = RESULTS_DIR / 'classification' / 'multi'
JUDGEMENTS_DIR = RESULTS_DIR / 'judgements'


CLASSIFICATION_ROOT = RESULTS_DIR / 'classification'
CLASSIFICATION_DIR = CLASSIFICATION_ROOT / 'single'
MULTI_CLASSIFICATION_DIR = CLASSIFICATION_ROOT / 'multi'

LABEL_ROOT = DATA_DIR / 'label'
LABEL_DIR = LABEL_ROOT / 'single'
MULTI_LABEL_DIR = LABEL_ROOT / 'multi'

ANNOTATION_ROOT = RESULTS_DIR / 'annotation'
ANNOTATION_DIR = ANNOTATION_ROOT / 'single'
MULTI_ANNOTATION_DIR = ANNOTATION_ROOT / 'multi'
MANUAL_DIR = ANNOTATION_DIR / 'manual'
JUDGE_DIR = ANNOTATION_DIR / 'judge'
LANGUAGE_DIR = RESULTS_DIR / 'language'

JUDGEMENTS_PATH = RESULTS_DIR / 'judgements.csv'


PLAN_PATH = DIALOGUE_DIR / 'plan.csv'
TURNS_PATH = DIALOGUE_DIR / 'turns.csv'
WITHHELD_PATH = DIALOGUE_DIR / 'withheld.csv'

DATA_DIRS = [DATA_DIR, PROCESS_DIR, ORIGINAL_DIR, BATCHES_DIR, RESULTS_DIR,
             ADAPTATION_DIR, DIALOGUE_DIR, JUDGEMENTS_DIR, LANGUAGE_DIR,
             LABEL_DIR, ANNOTATION_DIR, MANUAL_DIR, JUDGE_DIR,
             CLASSIFICATION_DIR]


class _NoDuplicates(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise SystemExit(
                    f'{key!r} appears more than once in the same block, at line '
                    f'{key_node.start_mark.line + 1}. YAML would keep only the '
                    f'last one.')
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_config(path):
    with open(path, encoding='utf-8') as file:
        return yaml.load(file, Loader=_NoDuplicates)


SETTINGS = read_config(SETTINGS_PATH)


SCENARIO_SETTINGS = read_config(SCENARIOS_PATH)


POLICY = read_config(JUDGE_PATH)

SEED = SETTINGS['seed']
DOMAINS = SETTINGS['domains']
TYPES = SETTINGS['types']
CUES = SETTINGS['cues']
ANSWERS = SETTINGS['answers']


PERMISSIVENESS = {'Refusal': 0, 'Compliance': 1}
AGE_BAND_LIMITS = SETTINGS['age_bands']
EXPLICIT_OPENER = SETTINGS['explicit_opener']
SAFETY = SETTINGS['safety']
LANGUAGE = SETTINGS['language']
METHODS = SETTINGS['methods']
DIALOGUE = SETTINGS['dialogue']

DATASETS = SETTINGS['datasets']
SOURCES = SETTINGS['sources']
SAFECHILD_COMMIT = SETTINGS['safechild_commit']

GENERATION = SETTINGS['generation']
PROVIDER_KEYS = SETTINGS['keys']
MODELS = SETTINGS['models']
JUDGES = SETTINGS['judges']
JUDGE = JUDGES['primary']

SCENARIOS = SCENARIO_SETTINGS['scenarios']


AGE_BANDS = list(AGE_BAND_LIMITS)


MINOR = 'Minor'
MINOR_BANDS = [band for band in AGE_BANDS if band != AGE_BANDS[-1]]
BANDS = AGE_BANDS + [MINOR]


def band_for_age(age):
    if age == '' or age is None:
        return ''
    age = int(age)
    for band, oldest in AGE_BAND_LIMITS.items():
        if age <= oldest:
            return band
    raise ValueError(f'Age {age} falls outside every band')


def measure_column(name):
    return name.lower().replace(' ', '_')


def expand_condition(condition):
    age = condition.get('age', '')
    opener = condition.get('opener')
    if opener is None:
        opener = EXPLICIT_OPENER.format(age=age) if age != '' else ''
    return {'name': condition['name'],
            'age': age,
            'band': condition.get('band', ''),
            'signal': condition['signal'],
            'cue': condition['cue'],
            'opener': opener}


CONDITIONS = [expand_condition(condition) for condition in SETTINGS['conditions']]
CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})


DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
DOMAIN_CODES = {name: code for code, name in DOMAIN_NAMES.items()}
TYPE_ANSWERS = {name: values['answers'] for name, values in TYPES.items()}
TYPE_CODES = {name: values['code'] for name, values in TYPES.items()}
CATEGORIES = SETTINGS['categories']
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)


AUTHORED = 'Author'
DATASET_NAMES = sorted({spec['name'] for spec in SOURCES.values()} | {AUTHORED})


DRAFTS_COLUMNS = ['source_id', 'dataset', 'domain', 'scenario_type', 'category',
                  'order', 'source_prompt', 'request']


DRAFTS_WRITTEN = [column for column in DRAFTS_COLUMNS if column != 'request']


BENCHMARK_COLUMNS = ['scenario_id', 'source_id', 'dataset', 'domain',
                     'scenario_type', 'category', 'request']


PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                  'signal', 'cue', 'opener', 'request', 'prompt',
                  'expected_answer']


RESPONSE_COLUMNS = ['model', 'prompt_id', 'replicate', 'error', 'blocked',
                    'truncated', 'response']


JUDGEMENT_COLUMNS = (['model', 'prompt_id', 'replicate', 'judge', 'policy',
                      'answer']
                     + [measure_column(name) for name in SAFETY]
                     + ['unreadable', 'error'])


BLOCKED = 'Blocked'


LANGUAGE_COLUMNS = (['model', 'prompt_id', 'replicate']
                    + [measure_column(name) for name in LANGUAGE])
assert len(set(LANGUAGE_COLUMNS)) == len(LANGUAGE_COLUMNS), \
    'duplicate language column, check config/settings.yml'


DIALOGUE_COLUMNS = ['dialogue_id', 'prompt_id', 'scenario_id', 'condition',
                    'band', 'model', 'opening_replicate', 'method', 'turn',
                    'role', 'text', 'expected_answer']


assert all(condition['band'] == band_for_age(condition['age'])
           for condition in CONDITIONS if condition['age'] != '')
assert all(condition['band'] in BANDS + [''] for condition in CONDITIONS)
assert all(condition['cue'] in CUES for condition in CONDITIONS)
assert all(condition['signal'] in ('Explicit', 'Implicit', 'None')
           for condition in CONDITIONS)
assert len(set(CONDITION_NAMES)) == len(CONDITIONS)
assert set(BANDS) >= {condition['band'] for condition in CONDITIONS
                      if condition['band']}
assert all(set(answers) == set(AGE_BANDS) for answers in TYPE_ANSWERS.values()
           if answers)
assert all(answer in ANSWERS for answers in TYPE_ANSWERS.values()
           for answer in answers.values())
assert set(PERMISSIVENESS) == set(ANSWERS)


assert all([PERMISSIVENESS[answers[band]] for band in AGE_BANDS]
           == sorted(PERMISSIVENESS[answer] for answer in answers.values())
           for answers in TYPE_ANSWERS.values() if answers)


assert len({(tuple(values['answers'][band] for band in AGE_BANDS),
             values['salience'])
            for values in TYPES.values() if values['answers']}) \
    == len([values for values in TYPES.values() if values['answers']])
assert all(values['salience'] in ('High', 'Low') for values in TYPES.values())
assert len(set(TYPE_CODES.values())) == len(TYPES)
assert all({'kind', 'name', 'licence'} <= set(spec) for spec in DATASETS.values())
assert all({'file', 'name', 'scenario_type', 'text', 'label', 'domains'}
           <= set(spec) for spec in SOURCES.values())
assert all(spec['scenario_type'] in TYPES for spec in SOURCES.values())
assert all(code in DOMAINS for spec in SOURCES.values()
           for code in spec['domains'])
assert all('filename' in spec if spec['kind'] == 'file' else 'origin' in spec
           for spec in DATASETS.values())
assert all(spec['kind'] in ('hub', 'url', 'file') for spec in DATASETS.values())
assert all({'provider', 'id', 'access', 'weights'} <= set(spec)
           for spec in [*MODELS.values(), *JUDGES.values()])
assert all(spec['access'] in ('api', 'local') for spec in MODELS.values())
assert all(spec['provider'] in PROVIDER_KEYS for spec in MODELS.values()
           if spec['access'] == 'api')
assert all(method in METHODS for method in DIALOGUE['methods'])
assert all(len(values) >= 2 for values in SAFETY.values())
assert len({len(spec['turns']) for spec in METHODS.values()}) == 1
