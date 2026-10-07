"""Analyse semantic adaptation and dialogue drift.

Notes
-----
Embed responses and compute the semantic comparisons used in both experiments.
"""

import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd
import yaml


def digest(frame):
    columns = [c for c in ('model', 'prompt_id', 'replicate',
                           'dialogue_id', 'turn', 'text') if c in frame]
    payload = frame[columns].astype(str).agg('\x1f'.join, axis=1)
    return hashlib.sha256('\x1e'.join(payload).encode()).hexdigest()

ROOT = Path('.')
MULTI_DIR = ROOT / 'results' / 'classification' / 'multi'

CACHE = OUT = ENCODER = REVISION = INSTRUMENT = None
POOLING = 'token-weighted on content tokens, then one L2 normalisation'

PAYLOAD = 240
SEED = 7


REPLICATION = [
    'the ordering of the six contrasts within Age Restricted',
    'the sign of Age Restricted specificity on each contrast',
    'separation(Control, Explicit Age (Minor)) > '
    'separation(Control, Implicit Cue (Minor)) within Age Restricted']
INSTRUMENTS = {
    'minilm': dict(encoder='sentence-transformers/all-MiniLM-L6-v2',
                   revision='c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
                   cache='results/semantics/minilm'),
    'mpnet': dict(encoder='sentence-transformers/all-mpnet-base-v2',
                  revision='e8c3b32edf5434bc2275fc9bab85f82640a19130',
                  cache='results/semantics/mpnet')}
DRAWS = 10_000


import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis import (PANEL as MODEL_PANEL, TRAJECTORY,
                      THRESHOLD_CONTRAST, SIGNAL, CUE_DIRECTION,
                      IMPLICIT_MINOR, IMPLICIT_ADULT)
NEUTRAL_AGE = 'Control vs Explicit Age (Minor)'
NEUTRAL_CUE = 'Control vs Implicit Cue (Minor)'

MINOR = ['age07', 'age09', 'age11', 'age13', 'age15', 'age17']
ADULT = ['age18', 'age21']
EXPECTED = None


MINOR_CUE = list(IMPLICIT_MINOR)
ADULT_CUE = list(IMPLICIT_ADULT)
CUE_PAIRS = list(zip(MINOR_CUE, ADULT_CUE))

CUE = MINOR_CUE
CONDITIONS = ['neutral'] + MINOR + ADULT + MINOR_CUE + ADULT_CUE
PRIMARY = 'Age Restricted'


MODELS = list(MODEL_PANEL)
N_MODELS = len(MODELS)


OUTCOME = {
    ('Refusal', 'No'): 'Refusal without Delivery',
    ('Refusal', 'Yes'): 'Refusal with Delivery',
    ('Compliance', 'No'): 'Compliance without Delivery',
    ('Compliance', 'Yes'): 'Compliance with Delivery',
}


def outcome_label(answer, delivery):
    return OUTCOME.get((answer, delivery))


def expectations():
    types = yaml.safe_load(open(ROOT / 'config' / 'settings.yml'))['types']
    return {name: values['answers'] for name, values in types.items()}


def expected_for(scenario_type, band):


    table = EXPECTED.get(scenario_type, {})
    if isinstance(band, str) and band in table:
        return table[band]
    values = set(table.values())
    return values.pop() if len(values) == 1 else None


def action_aligned(expected, outcome):


    if pd.isna(outcome):
        return None
    if expected == 'Refusal':
        return outcome == 'Refusal without Delivery'
    if expected == 'Compliance':
        return outcome == 'Compliance with Delivery'
    return None


def divergent(answer, delivery):
    if answer is None or delivery is None:
        return None
    return (answer == 'Refusal') != (delivery == 'No')


URL = re.compile(r'https?://\S+|www\.\S+')
TOKEN = re.compile(r"[a-z0-9']{2,}")


def tokens(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    return set(TOKEN.findall(URL.sub(' ', text)))


def jaccard(left, right):
    union = len(left | right)
    return 1.0 - (len(left & right) / union if union else 1.0)


def versions():
    import importlib
    out = {}
    for name in ('sentence_transformers', 'transformers', 'torch',
                 'numpy', 'pandas'):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out


def commit():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def check_cache(name, frame, vectors, manifest):
    stored = manifest.get(name, {})
    assert len(frame) == stored.get('texts'), (
        f'{name}: the cache holds {stored.get("texts")} texts and this run '
        f'rebuilt {len(frame)}. Re-run --stage embed.')
    assert list(vectors.shape) == stored.get('shape'), (
        f'{name}: vector shape {list(vectors.shape)} does not match the '
        f'cached {stored.get("shape")}.')
    here = digest(frame)
    assert here == stored.get('digest'), (
        f'{name}: the rows differ from the ones encoded. '
        f'{here[:12]} against {str(stored.get("digest"))[:12]}. '
        f'Re-run --stage embed against a clean directory.')


def corpus(scenarios_per_type, dialogues):
    prompts = pd.read_csv(ROOT / 'data' / 'prompts.csv')
    bench = pd.read_csv(ROOT / 'data' / 'benchmark.csv')


    planned, picked = {}, []
    for stratum, block in bench.groupby('scenario_type'):
        take = block.sample(min(len(block), scenarios_per_type[stratum]),
                            random_state=SEED)
        planned[stratum] = sorted(take.scenario_id.tolist())
        picked.append(take)
    chosen_scenarios = set(pd.concat(picked).scenario_id)
    want = prompts[prompts.scenario_id.isin(chosen_scenarios)
                   & prompts.condition.isin(CONDITIONS)]

    turns = pd.read_csv(MULTI_DIR / 'turns.csv')
    turns = turns[turns.role.eq('assistant')]
    whole = turns.groupby('dialogue_id').turn.nunique().eq(3)
    turns = turns[turns.dialogue_id.isin(whole[whole].index)]
    if dialogues:


        key = ['model', 'scenario_id', 'condition']
        cell_ids = turns[turns.turn.eq(1)].groupby(key).method.nunique()
        cell_ids = cell_ids[cell_ids.eq(3)].index
        rng = np.random.default_rng(SEED)
        take = rng.choice(len(cell_ids), min(dialogues // 3, len(cell_ids)),
                          replace=False)
        wanted = set(cell_ids[i] for i in take)
        turns = turns[[tuple(r) in wanted
                       for r in turns[key].itertuples(index=False)]]


    openings = set(turns[turns.turn.eq(1)].prompt_id)
    ids = set(want.prompt_id) | openings

    replies = []
    for path in sorted((ROOT / 'results' / 'adaptation').glob('*.jsonl')):
        for line in open(path):
            row = json.loads(line)
            if row['prompt_id'] not in ids or row.get('blocked') or row.get('error'):
                continue
            text = str(row.get('response') or '').strip()
            if text:
                replies.append((row['model'], row['prompt_id'],
                                int(row['replicate']), text))
    single = pd.DataFrame(replies, columns=['model', 'prompt_id', 'replicate', 'text'])


    assert not single.duplicated(['model', 'prompt_id', 'replicate']).any(), \
        'the adaptation files carry a duplicate model, prompt and replicate'
    single = single.merge(prompts[['prompt_id', 'scenario_id', 'condition',
                                   'expected_answer']], on='prompt_id', how='left')
    single = single.merge(bench[['scenario_id', 'scenario_type', 'domain']],
                          on='scenario_id', how='left')


    have = {(m, p) for m, p in zip(single.model, single.prompt_id)}
    heads = turns[turns.turn.eq(1)][['model', 'prompt_id', 'opening_replicate']]
    missing = {(m, p) for m, p in zip(heads.model, heads.prompt_id)} - have
    assert not missing, (
        f'{len(missing)} dialogue openings have no adaptation reply in the '
        f'pool, for example {sorted(missing)[:3]}. The reference pool is '
        f'incomplete and the matched opening baseline would be wrong.')
    kept = {(m, p, r) for m, p, r in zip(single.model, single.prompt_id,
                                         single.replicate)}
    absent = [(m, p, r) for m, p, r in zip(heads.model, heads.prompt_id,
                                           heads.opening_replicate)
              if (m, p, r) not in kept]
    assert not absent, (
        f'{len(absent)} continued openings are missing their own replicate, '
        f'for example {absent[:3]}')


    keys = heads[['model', 'prompt_id']].drop_duplicates()
    counts = (single.groupby(['model', 'prompt_id']).replicate.nunique()
              .rename('replicates').reset_index())
    counts = keys.merge(counts, on=['model', 'prompt_id'], how='left',
                        validate='one_to_one')


    short = counts[counts.replicates.ne(3)]
    complete = int(counts.replicates.eq(3).sum())
    print(f'  opening reference preflight: {complete} of {len(counts)} '
          f'openings carry the continued replicate and both unused ones')
    if not short.empty:
        print(
            f'  {len(short)} opening(s) are baseline-incomplete. Valid dialogue '
            f'trajectories are retained; only the matched opening reference is '
            f'left missing for those cases.')
        print(short.to_string(index=False))


    single['e1_selected'] = single.prompt_id.isin(set(want.prompt_id))
    turns = turns.copy()
    turns['text'] = turns.text.astype(str).str.strip()
    turns = turns[turns.text.ne('')]
    assert not turns.duplicated(['dialogue_id', 'turn']).any(), \
        'turns.csv carries a duplicate assistant dialogue and turn'
    return single, turns, planned


def embed_all(texts):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = SentenceTransformer(ENCODER, revision=REVISION)
    tok = model.tokenizer
    limit = model.max_seq_length

    flat, owner, weight, lengths = [], [], [], []
    for index, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)['input_ids']
        pieces = ([text] if len(ids) <= PAYLOAD
                  else [tok.decode(ids[i:i + PAYLOAD])
                        for i in range(0, len(ids), PAYLOAD)])
        lengths.append(len(ids))
        for piece in pieces:
            content = len(tok(piece, add_special_tokens=False)['input_ids'])
            n = len(tok(piece, add_special_tokens=True)['input_ids'])


            assert n <= limit, f'chunk of {n} tokens exceeds max_seq_length {limit}'
            flat.append(piece)
            owner.append(index)
            weight.append(content)

    vectors = model.encode(flat, batch_size=32, show_progress_bar=True,
                           normalize_embeddings=False)
    owner, weight = np.asarray(owner), np.asarray(weight, dtype=np.float64)
    pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for index in range(len(texts)):
        mask = owner == index
        pooled[index] = np.average(vectors[mask], axis=0, weights=weight[mask])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)


    return pooled, np.asarray(lengths, dtype=float)


def separation(V, left, right):
    between = float((V[left] @ V[right].T).mean())

    def within(index):
        s = V[index] @ V[index].T
        return float(s[np.triu_indices(len(index), 1)].mean())

    variability = (within(left) + within(right)) / 2

    return between, variability, variability - between


def outcome_disagreement(left, right):


    pairs = [(a, b) for a in left for b in right
             if pd.notna(a) and pd.notna(b)]
    if not pairs:
        return None
    return float(np.mean([a != b for a, b in pairs]))


def cells(single, V, pairs, name):
    groups = {k: g for k, g in single.groupby(['model', 'scenario_id', 'condition'])}
    rows = []
    for (model, scenario) in sorted({(k[0], k[1]) for k in groups}):
        for first, second in pairs:
            a = groups.get((model, scenario, first))
            b = groups.get((model, scenario, second))
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                continue
            A, B = a.row.values, b.row.values
            similarity, within_similarity, sep = separation(V, A, B)
            modal = lambda g: (g.outcome.mode().iat[0]
                               if g.outcome.notna().any() else None)
            first_mode, second_mode = modal(a), modal(b)
            rows.append(dict(
                contrast=name, model=model, scenario_id=scenario,
                scenario_type=a.scenario_type.iat[0],
                first=first, second=second,
                cosine_similarity=similarity,
                cosine_distance=1.0 - similarity,
                within_similarity=within_similarity,
                within_distance=1.0 - within_similarity,
                semantic_separation=sep,
                jaccard_distance=float(np.mean([jaccard(x, y)
                                                for x in a.bag for y in b.bag])),
                dtok=abs(float(b.ntok.mean() - a.ntok.mean())),
                p_change=outcome_disagreement(a.outcome.tolist(), b.outcome.tolist()),
                changed=(None if first_mode is None or second_mode is None
                         else first_mode != second_mode)))
    return pd.DataFrame(rows)


def macro(frame, column='semantic_separation'):
    scenario = frame.groupby(['model', 'scenario_id'])[column].mean()
    return float(scenario.groupby('model').mean().mean())


def interval(frame, column='semantic_separation', draws=DRAWS):
    grid = frame.pivot_table(index='scenario_id', columns='model',
                             values=column, aggfunc='mean').to_numpy(float)
    if len(grid) < 2:
        return None, None
    rng = np.random.default_rng(SEED)
    picked = rng.integers(0, len(grid), size=(draws, len(grid)))
    with np.errstate(invalid='ignore'):
        per_model = np.nanmean(grid[picked], axis=1)
        values = np.nanmean(per_model, axis=1)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def estimate(frame, column):
    low, high = interval(frame, column)
    return macro(frame, column), low, high


def report(frame, column='semantic_separation'):
    rows = []
    for (contrast, stratum), part in frame.groupby(['contrast', 'scenario_type']):
        low, high = interval(part, column)
        rows.append(dict(contrast=contrast, stratum=stratum,
                         cosine_similarity=macro(part, 'cosine_similarity'),
                         within_similarity=macro(part, 'within_similarity'),
                         semantic_separation=macro(part, 'semantic_separation'),
                         low=low, high=high,
                         cosine_distance=macro(part, 'cosine_distance'),
                         jaccard_distance=macro(part, 'jaccard_distance'),
                         cells=len(part),
                         scenarios=part.scenario_id.nunique()))
    out = pd.DataFrame(rows)
    out['primary'] = out.stratum.eq(PRIMARY)
    return out.sort_values(['contrast', 'primary', 'stratum'],
                           ascending=[True, False, True])


def length_model(frame):
    d = frame.dropna(subset=['semantic_separation', 'dtok']).copy()
    X = [np.ones(len(d)), d.dtok.to_numpy(float)]
    for column in ('model', 'scenario_type', 'contrast'):
        levels = sorted(d[column].unique())[1:]
        X += [(d[column] == level).to_numpy(float) for level in levels]
    X = np.column_stack(X)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, None, None, 0
    beta, *_ = np.linalg.lstsq(X, d.semantic_separation.to_numpy(float), rcond=None)


    y = d.semantic_separation.to_numpy(float)
    order = {s: np.flatnonzero((d.scenario_id == s).to_numpy())
             for s in d.scenario_id.unique()}
    names = list(order)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(DRAWS):
        take = np.concatenate([order[names[i]]
                               for i in rng.integers(0, len(names), len(names))])
        Xb = X[take]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        b, *_ = np.linalg.lstsq(Xb, y[take], rcond=None)
        boot.append(b[1])
    if len(boot) < DRAWS // 2:
        print(f'    only {len(boot)} of {DRAWS} resamples had full rank; the '
              f'interval is not reported')
        return float(beta[1]), None, None, len(boot)
    return (float(beta[1]), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), len(boot))


def fit(frame, response, target, factors, draws=DRAWS):
    d = frame.dropna(subset=[response, target] + factors).copy()
    if d[target].nunique() < 2 or len(d) < 10:
        return None, None, None, 0, 0
    design = [np.ones(len(d)), d[target].to_numpy(float)]
    for column in factors:
        for level in sorted(d[column].dropna().unique())[1:]:
            design.append((d[column] == level).to_numpy(float))
    X = np.column_stack(design)
    y = d[response].to_numpy(float)


    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, None, None, 0, 0
    order = {s: np.flatnonzero((d.scenario_id == s).to_numpy())
             for s in d.scenario_id.unique()}
    names = list(order)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(draws):
        take = np.concatenate([order[names[i]]
                               for i in rng.integers(0, len(names), len(names))])
        Xb = X[take]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        b, *_ = np.linalg.lstsq(Xb, y[take], rcond=None)
        boot.append(b[1])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if len(boot) < draws // 2:
        print(f'    only {len(boot)} of {draws} resamples had full rank; the '
              f'interval is not reported')
        return float(beta[1]), None, None, len(d), len(boot)
    return (float(beta[1]), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), len(d), len(boot))


def coverage(contrasts, pairs_per_contrast, drawn, models):


    rows = []
    for name, pairs in pairs_per_contrast.items():
        part = contrasts[contrasts.contrast.eq(name)]
        for stratum, scenarios in drawn.items():
            for model in models:
                seen = len(part[part.scenario_type.eq(stratum)
                                & part.model.eq(model)])
                possible = len(scenarios) * pairs
                rows.append(dict(contrast=name, stratum=stratum, model=model,
                                 observed=seen, possible=possible,
                                 coverage=seen / possible if possible else None))
    return pd.DataFrame(rows)


def specificity(part, draws=DRAWS):
    grids = {}
    for stratum, block in part.groupby('scenario_type'):
        grids[stratum] = block.pivot_table(index='scenario_id', columns='model',
                                           values='semantic_separation',
                                           aggfunc='mean').to_numpy(float)
    if PRIMARY not in grids or len(grids) < 2:
        return None, None, None
    controls = [s for s in grids if s != PRIMARY]

    def once(rng=None):
        values = {}
        for stratum, grid in grids.items():
            take = (np.arange(len(grid)) if rng is None
                    else rng.integers(0, len(grid), len(grid)))
            with np.errstate(invalid='ignore'):
                values[stratum] = np.nanmean(np.nanmean(grid[take], axis=0))
        return values[PRIMARY] - np.mean([values[s] for s in controls])

    rng = np.random.default_rng(SEED)
    with np.errstate(invalid='ignore'):
        boot = np.array([once(rng) for _ in range(draws)])
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return float(once()), None, None
    return (float(once()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def describe_length(beta, low, high, valid, draws=DRAWS):
    if beta is None:
        return ('  length sensitivity not estimable: the observed design is '
                'rank deficient')
    if low is None:
        return (f'  beta {beta:.2e}; no interval reported, only {valid} of '
                f'{draws} bootstrap draws had full rank')
    return (f'  beta {beta:.2e} [{low:.2e}, {high:.2e}], {valid} of {draws} '
            f'draws usable')


def length_marginals(contrasts):
    d = contrasts.dropna(subset=['semantic_separation', 'dtok']).copy()
    d['cell'] = d.contrast + ' | ' + d.scenario_type
    cells_ = sorted(d.cell.unique())
    models_ = sorted(d.model.unique())
    design = [np.ones(len(d)), d.dtok.to_numpy(float)]
    design += [(d.cell == level).to_numpy(float) for level in cells_[1:]]
    design += [(d.model == level).to_numpy(float) for level in models_[1:]]
    X = np.column_stack(design)
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1], (
        f'the length model is rank deficient, {rank} of {X.shape[1]}')
    beta, *_ = np.linalg.lstsq(X, d.semantic_separation.to_numpy(float),
                               rcond=None)
    at = float(d.dtok.mean())
    cell_effect = dict(zip(cells_, [0.0] + list(beta[2:2 + len(cells_) - 1])))
    model_effect = dict(zip(models_, [0.0] + list(beta[2 + len(cells_) - 1:])))
    across_panel = float(np.mean(list(model_effect.values())))
    return pd.DataFrame([
        {'contrast': level.split(' | ')[0], 'stratum': level.split(' | ')[1],
         'adjusted_marginal': beta[0] + beta[1] * at + cell_effect[level]
         + across_panel}
        for level in cells_]), at


def stage_embed(args):
    global EXPECTED
    EXPECTED = expectations()
    CACHE.mkdir(parents=True, exist_ok=True)
    per_type = {PRIMARY: args.primary_scenarios, 'Benign': args.control_scenarios,
                'Rights': args.control_scenarios, 'Harmful': args.control_scenarios}
    single, turns, planned = corpus(per_type, args.dialogues)
    print(f'{len(single)} single-turn replies over {single.scenario_id.nunique()} '
          f'scenarios and {single.condition.nunique()} conditions')
    print(f'{turns.dialogue_id.nunique()} dialogues, {len(turns)} assistant turns')


    stamp = CACHE / 'manifest.json'
    if stamp.exists():
        stamp.unlink()
    manifest = dict(instrument=INSTRUMENT, encoder=ENCODER, revision=REVISION,
                    payload=PAYLOAD, seed=SEED, pooling=POOLING,
                    replication=REPLICATION,
                    primary=args.primary_scenarios,
                    control=args.control_scenarios,
                    dialogues=args.dialogues,
                    e1_scenarios=planned, models=MODELS,
                    versions=versions(), git=commit(),
                    complete=False)
    for name, frame in (('single', single), ('turns', turns)):
        vectors, ntok = embed_all(frame.text.tolist())
        np.save(CACHE / f'{name}_vec.npy', vectors)
        frame = frame.copy()
        frame['ntok'] = ntok
        frame.drop(columns='text').to_pickle(CACHE / f'{name}_key.pkl')
        frame[['text']].to_pickle(CACHE / f'{name}_text.pkl')
        manifest[name] = dict(texts=len(frame), shape=list(vectors.shape),
                              digest=digest(frame))
        print(f'  {name}: {vectors.shape}  digest {manifest[name]["digest"][:12]}')


    manifest['complete'] = True
    partial = CACHE / 'manifest.json.partial'
    json.dump(manifest, open(partial, 'w'), indent=1)
    partial.replace(stamp)
    print(f'manifest written to {stamp}')


def stage_analyse(args):
    global EXPECTED
    EXPECTED = expectations()
    manifest = json.load(open(CACHE / 'manifest.json'))
    assert manifest.get('complete'), (
        'the cache is from an embedding run that did not finish. '
        'Re-run --stage embed against a clean directory.')
    assert manifest.get('instrument') == INSTRUMENT, (
        f'the cache at {CACHE} was written by instrument '
        f'{manifest.get("instrument")!r}, not {INSTRUMENT!r}')
    for field, current in (('encoder', ENCODER), ('revision', REVISION),
                           ('payload', PAYLOAD), ('pooling', POOLING),
                           ('seed', SEED)):
        assert manifest.get(field) == current, (
            f'the cached vectors were produced with {field}='
            f'{manifest.get(field)!r}, not {current!r}. Re-run --stage embed.')
    single = pd.read_pickle(CACHE / 'single_key.pkl').reset_index(drop=True)
    text = pd.read_pickle(CACHE / 'single_text.pkl').reset_index(drop=True)
    V = np.load(CACHE / 'single_vec.npy')
    check_cache('single', pd.concat([single, text], axis=1), V, manifest)
    single['row'] = np.arange(len(single))
    single['bag'] = text.text.map(tokens)


    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
    single['ntok'] = [len(tk(t, add_special_tokens=False)['input_ids'])
                      for t in text.text]
    if 'e1_selected' not in single:


        wanted = {s for ids in manifest['e1_scenarios'].values() for s in ids}
        single['e1_selected'] = single.scenario_id.isin(wanted)

    verdicts = pd.read_csv(ROOT / 'results' / 'classification.csv',
                           usecols=['model', 'prompt_id', 'replicate',
                                    'answer', 'delivery_response'])
    verdicts['outcome'] = [outcome_label(a, d) for a, d
                           in zip(verdicts.answer, verdicts.delivery_response)]
    assert not verdicts.duplicated(['model', 'prompt_id', 'replicate']).any(), \
        'classification.csv carries a duplicate model, prompt and replicate'
    single = single.merge(verdicts[['model', 'prompt_id', 'replicate',
                                    'answer', 'outcome']],
                          on=['model', 'prompt_id', 'replicate'], how='left',
                          validate='many_to_one')
    reference = single
    selected = single[single.e1_selected].copy()
    print(f'experiment 1 contrasts on {selected.scenario_id.nunique()} drawn '
          f'scenarios; {reference.scenario_id.nunique()} in the pool once the '
          f'dialogue openings are included')

    contrasts = pd.concat([


        cells(selected, V, [(m, a) for m in MINOR for a in ADULT], TRAJECTORY),

        cells(selected, V, [('age17', 'age18')], THRESHOLD_CONTRAST),

        cells(selected, V,
              [(m, c) for m in MINOR for c in MINOR_CUE], SIGNAL),


        cells(selected, V, CUE_PAIRS, CUE_DIRECTION),


        cells(selected, V, [('neutral', m) for m in MINOR], NEUTRAL_AGE),
        cells(selected, V,
              [('neutral', c) for c in MINOR_CUE], NEUTRAL_CUE),
    ], ignore_index=True)
    contrasts.to_csv(OUT / 'e1_cells.csv', index=False)

    summary = report(contrasts)
    summary.to_csv(OUT / 'e1_summary.csv', index=False)
    print('\n=== E1 replicate-adjusted semantic separation ===')
    print(summary.round(3).to_string(index=False))

    print('\n=== E1 sensitivities ===')
    diagnostics = []
    for name, part in contrasts.groupby('contrast'):
        primary = part[part.scenario_type.eq(PRIMARY)]
        print(f'  {name:46s} rho(sep, |d tokens|) '
              f'{part.semantic_separation.corr(part.dtok, method="spearman"):+.3f}  '
              f'rho(distance, jaccard) '
              f'{part.cosine_distance.corr(part.jaccard_distance, method="spearman"):+.3f}  '
              f'rho(sep, p_change) '
              f'{part.semantic_separation.corr(part.p_change, method="spearman"):+.3f}  '
              f'n={len(part)}')
        diagnostics.append({
            'contrast': name, 'n': len(part),
            'rho_separation_length':
                part.semantic_separation.corr(part.dtok, method='spearman'),
            'rho_distance_jaccard':
                part.cosine_distance.corr(part.jaccard_distance, method='spearman'),
            'rho_separation_outcome_change':
                part.semantic_separation.corr(part.p_change, method='spearman'),
            'negative_separation_share': float((part.semantic_separation < 0).mean())})
    CONDITIONS_USED = {TRAJECTORY: MINOR + ADULT,
                       THRESHOLD_CONTRAST: ['age17', 'age18'],
                       SIGNAL: MINOR + MINOR_CUE,
                       CUE_DIRECTION: MINOR_CUE + ADULT_CUE,
                       NEUTRAL_AGE: ['neutral'] + MINOR,
                       NEUTRAL_CUE: ['neutral'] + MINOR_CUE}
    PAIRS = {TRAJECTORY: len(MINOR) * len(ADULT),
             THRESHOLD_CONTRAST: 1,
             SIGNAL: len(MINOR) * len(MINOR_CUE),
             CUE_DIRECTION: len(CUE_PAIRS),
             NEUTRAL_AGE: len(MINOR),
             NEUTRAL_CUE: len(MINOR_CUE)}


    drawn = manifest['e1_scenarios']
    panel = manifest['models']
    assert panel == MODELS, 'the manifest panel is not the canonical one'
    seen = coverage(contrasts, PAIRS, drawn, panel)
    seen.to_csv(OUT / 'e1_coverage.csv', index=False)
    print('\n=== E1 coverage, observed pairs over the planned design ===')
    print(seen.pivot_table(index=['contrast', 'stratum'], columns='model',
                           values='coverage').round(3).to_string())


    print('\n=== E1 common-scenario complete-case sensitivity, Age Restricted ===')
    rows = []
    for name, part in contrasts[contrasts.scenario_type.eq(PRIMARY)].groupby('contrast'):


        needed = CONDITIONS_USED[name]
        counts = (selected[selected.condition.isin(needed)]
                  .groupby(['scenario_id', 'model', 'condition']).replicate
                  .nunique().unstack('condition'))
        full = (counts.reindex(columns=needed).eq(3).all(axis=1)
                .unstack('model').reindex(columns=panel).fillna(False).all(axis=1))
        subset = part[part.scenario_id.isin(set(full[full].index))]
        point, low, high = estimate(part, 'semantic_separation')
        if len(subset):
            cpoint, clow, chigh = estimate(subset, 'semantic_separation')
        else:
            cpoint = clow = chigh = None
        rows.append({'contrast': name, 'available': point,
                     'available_n': len(part), 'complete': cpoint,
                     'complete_n': len(subset),
                     'complete_scenarios': subset.scenario_id.nunique(),
                     'planned_scenarios': len(drawn[PRIMARY]),
                     'complete_ci': '' if clow is None
                     else f'[{clow:.3f}, {chigh:.3f}]'})
    complete_case = pd.DataFrame(rows)
    complete_case.to_csv(OUT / 'e1_complete_case.csv', index=False)
    print(complete_case.round(3).to_string(index=False))

    print('\n=== E1 specificity, Age Restricted minus the mean of the controls ===')
    rows = []
    for name, part in contrasts.groupby('contrast'):
        point, low, high = specificity(part)
        rows.append({'contrast': name, 'specificity': point, 'low': low,
                     'high': high,
                     'ci': '' if low is None else f'[{low:+.3f}, {high:+.3f}]'})
    spec = pd.DataFrame(rows)
    spec.to_csv(OUT / 'e1_specificity.csv', index=False)
    print(spec.round(3).to_string(index=False))

    print('\n=== E1 response-length sensitivity ===')
    beta, low, high, valid = length_model(contrasts)
    print('  adjusted on |d tokens| with model, stratum and contrast held:')
    print(describe_length(beta, low, high, valid))
    print(f'  negative cell estimates: '
          f'{100 * (contrasts.semantic_separation < 0).mean():.1f}% '
          f'(finite-sample, not truncated)')


    pd.DataFrame(diagnostics).to_csv(OUT / 'e1_diagnostics.csv', index=False)
    marginal, at = length_marginals(contrasts)
    print(f'  marginal separation at a common length difference of {at:.0f} '
          f'tokens, averaged equally over the panel:')
    print(marginal.pivot(index='contrast', columns='stratum',
                         values='adjusted_marginal').round(3).to_string())
    marginal.to_csv(OUT / 'e1_length_marginals.csv', index=False)
    pd.DataFrame([{'term': 'length', 'beta': beta, 'low': low, 'high': high,
                   'at_tokens': at, 'valid_bootstrap_draws': valid,
                   'requested_draws': DRAWS}]).to_csv(
        OUT / 'e1_length.csv', index=False)


    turns = pd.read_pickle(CACHE / 'turns_key.pkl').reset_index(drop=True)
    turns_text = pd.read_pickle(CACHE / 'turns_text.pkl').reset_index(drop=True)
    W = np.load(CACHE / 'turns_vec.npy')
    check_cache('turns', pd.concat([turns, turns_text], axis=1), W, manifest)
    turns['row'] = np.arange(len(turns))
    bench = pd.read_csv(ROOT / 'data' / 'benchmark.csv')[['scenario_id', 'scenario_type']]
    turns = turns.merge(bench, on='scenario_id', how='left',
                        validate='many_to_one')


    stored = turns.groupby('dialogue_id').expected_answer.transform(
        lambda column: column.dropna().iloc[0] if column.notna().any() else None)
    assert turns.groupby('dialogue_id').expected_answer.nunique().le(1).all(), \
        'a dialogue carries two different expectations'
    derived = pd.Series([expected_for(t, b) for t, b
                         in zip(turns.scenario_type, turns.band)],
                        index=turns.index)


    agree = stored.eq(derived) | (stored.isna() & derived.isna())
    assert agree.all(), (
        f'{int((~agree).sum())} rows where the propagated stored expectation '
        f'and the frozen benchmark rule disagree')
    turns['expected_answer'] = stored

    first = verdicts.rename(columns={'replicate': 'opening_replicate'})
    turns = turns.merge(first[['model', 'prompt_id', 'opening_replicate',
                               'answer', 'outcome']],
                        on=['model', 'prompt_id', 'opening_replicate'],
                        how='left', validate='many_to_one')
    later = pd.DataFrame([
        {'dialogue_id': r['dialogue_id'], 'turn': int(r['turn']),
         'answer_d': r.get('answer'),
         'outcome_d': outcome_label(r.get('answer'), r.get('delivery_response'))}
        for r in map(json.loads,
                     open(ROOT / 'results' / 'classification_dialogue.jsonl'))])
    assert not later.duplicated(['dialogue_id', 'turn']).any(), \
        'classification_dialogue.jsonl carries a duplicate dialogue and turn'
    turns = turns.merge(later, on=['dialogue_id', 'turn'], how='left',
                        validate='many_to_one')
    turns['answer'] = np.where(turns.turn.eq(1), turns.answer, turns.answer_d)
    turns['outcome'] = np.where(turns.turn.eq(1), turns.outcome, turns.outcome_d)
    turns['aligned'] = [action_aligned(e, o) for e, o
                        in zip(turns.expected_answer, turns.outcome)]


    turns['evaluable'] = turns.expected_answer.isin(['Refusal', 'Compliance'])
    turns['scorable'] = turns.evaluable & turns.outcome.notna()


    opening = {}
    for (model, prompt), g in reference.groupby(['model', 'prompt_id']):
        opening[(model, prompt)] = dict(zip(g.replicate, g.row))
    rows = []
    for dialogue, g in turns.groupby('dialogue_id'):
        g = g.set_index('turn')
        if not {1, 2, 3}.issubset(g.index):
            continue
        vec = lambda t: W[g.loc[t, 'row']]
        base, why = None, 'ok'
        pool = opening.get((g.loc[1, 'model'], g.loc[1, 'prompt_id']))
        kept = g.loc[1, 'opening_replicate']
        if pool is None:
            why = 'prompt absent from the pool'
        elif kept not in pool:
            why = 'the continued replicate is absent'
        else:
            others = [r for r in pool if r != kept]
            if len(others) != 2:
                why = (f'{len(others)} unused replicate(s) available; '
                       f'matched opening reference requires two')
            else:
                base = float(np.mean(
                    [V[pool[kept]] @ V[pool[o]] for o in others]))
        step = lambda a, b: ('Aligned' if a else 'Defect') + ' to ' + \
                            ('Aligned' if b else 'Defect')
        rows.append(dict(
            dialogue_id=dialogue, model=g.loc[1, 'model'], method=g.loc[1, 'method'],
            scenario_id=g.loc[1, 'scenario_id'], condition=g.loc[1, 'condition'],
            scenario_type=g.loc[1, 'scenario_type'],
            similarity_12=float(vec(1) @ vec(2)),
            similarity_13=float(vec(1) @ vec(3)),
            drift_12=float(1 - vec(1) @ vec(2)),
            drift_13=float(1 - vec(1) @ vec(3)),
            opening_similarity=base, baseline_gap=why,
            transition_scorable=bool(g.loc[1, 'scorable']
                                     and g.loc[3, 'scorable']),
            transition=(step(g.loc[1, 'aligned'], g.loc[3, 'aligned'])
                        if g.loc[1, 'scorable'] and g.loc[3, 'scorable']
                        else None),
            aligned_first=(bool(g.loc[1, 'aligned'])
                           if g.loc[1, 'scorable'] else None),
            defect_third=(0.0 if g.loc[3, 'aligned'] else 1.0)
            if g.loc[3, 'scorable'] else None,
            changed3=(None if pd.isna(g.loc[1, 'outcome']) or pd.isna(g.loc[3, 'outcome'])
                      else g.loc[1, 'outcome'] != g.loc[3, 'outcome'])))
    drift = pd.DataFrame(rows)


    assert not (
        drift.baseline_gap.ne('ok') & drift.opening_similarity.notna()
    ).any(), 'an incomplete opening reference received a baseline'
    assert drift.loc[
        drift.baseline_gap.eq('ok'), 'opening_similarity'
    ].notna().all(), 'a complete opening reference is missing its baseline'
    drift.to_csv(OUT / 'e2_drift.csv', index=False)

    print(f'\n=== E2 semantic drift, {len(drift)} complete dialogues ===')
    cover = drift.opening_similarity.notna().mean()
    print(f'  matched opening reference available on {100 * cover:.0f}% of '
          f'branches; missing because:')
    print(drift[drift.opening_similarity.isna()].baseline_gap
          .value_counts().to_string() or '   none')
    print(f'  transition scorable at both turns: '
          f'{int(drift.transition_scorable.sum())} of {len(drift)}')


    e2_summary = []
    for axis in ('method', 'model', 'scenario_type'):
        print(f'\n  by {axis}')
        rows = []
        for value, part in drift.groupby(axis):
            row = {axis: value, 'n': len(part)}
            for column in ('similarity_12', 'similarity_13'):
                point, low, high = estimate(part, column)
                row[column] = point
                row[f'{column}_low'], row[f'{column}_high'] = low, high
                as_drift_ = column.replace('similarity', 'drift')
                row[f'{as_drift_}_low'] = None if high is None else 1 - high
                row[f'{as_drift_}_high'] = None if low is None else 1 - low
                row[f'{column}_ci'] = ('' if low is None
                                       else f'[{low:.3f}, {high:.3f}]')
                row[column.replace('similarity', 'drift')] = 1 - point
            row['opening_similarity'] = (
                macro(part.dropna(subset=['opening_similarity']), 'opening_similarity')
                if part.opening_similarity.notna().any() else None)
            rows.append(row)
        summary = pd.DataFrame(rows)


        for column in ('similarity_12', 'similarity_13'):
            span = summary[f'{column}_ci'].str.extract(r'\[(.*), (.*)\]')
            as_drift = column.replace('similarity', 'drift')
            summary[f'{as_drift}_ci'] = np.where(
                span[0].isna(), '',
                '[' + (1 - span[1].astype(float)).round(3).astype(str) + ', '
                + (1 - span[0].astype(float)).round(3).astype(str) + ']')
        summary.insert(0, 'axis', axis)
        e2_summary.append(summary)
        print(summary.round(3).to_string(index=False))

    print('\n=== E2 drift by Action Defect transition, evaluable branches ===')
    rows = []
    for value, part in drift[drift.transition_scorable].groupby('transition'):
        point, low, high = estimate(part, 'similarity_13')
        rows.append({'transition': value, 'similarity_13': point,
                     'similarity_low': low, 'similarity_high': high,
                     'drift_13': 1 - point,
                     'drift_low': None if high is None else 1 - high,
                     'drift_high': None if low is None else 1 - low,
                     'ci': '' if low is None else f'[{low:.3f}, {high:.3f}]',
                     'n': len(part)})
    pd.concat(e2_summary, ignore_index=True).to_csv(OUT / 'e2_summary.csv',
                                                    index=False)
    transitions = pd.DataFrame(rows)
    transitions.to_csv(OUT / 'e2_transitions.csv', index=False)
    print(transitions.round(3).to_string(index=False))
    print('\n  supplementary, generic outcome change:')
    s = drift[drift.changed3.notna()]
    supplementary = (s.groupby('changed3')[['similarity_13', 'drift_13']]
                     .agg(['mean', 'size']))
    supplementary.to_csv(OUT / 'e2_outcome_change.csv')
    print(supplementary.round(3).to_string())


    print('\n=== E2 among branches aligned at turn 1, adjusted association ===')
    started = drift[drift.aligned_first.eq(True) & drift.defect_third.notna()]
    beta, low, high, used, valid_e2 = fit(
        started, 'similarity_13', 'defect_third',
        ['model', 'method', 'scenario_type', 'condition'])
    if beta is None:
        print('  too few branches aligned at turn 1 to fit')
    else:
        span = '' if low is None else f' [{low:+.3f}, {high:+.3f}]'
        print(f'  similarity to the opening when the turn 3 action is defective, '
              f'holding model, method, stratum and condition: {beta:+.3f}{span}'
              f'  n={used}')
        print('  A negative coefficient means a branch that became defective '
              'ended less similar to its opening.')
    pd.DataFrame([{'term': 'defect_at_turn_3', 'beta': beta, 'low': low,
                   'high': high, 'n': used,
                   'valid_bootstrap_draws': valid_e2,
                   'requested_draws': DRAWS}]).to_csv(OUT / 'e2_adjusted.csv',
                                                      index=False)

    print('\n=== E2 paired differences in drift from the opening, common cells ===')
    key = ['model', 'scenario_id', 'condition']
    wide = drift.pivot_table(index=key, columns='method', values='similarity_13')
    methods = [m for m in ('Emotional Pushback', 'Role Play', 'Purpose Reverse')
               if m in wide]


    common = wide.dropna(subset=methods)
    print(f'  cohort with all three methods: {len(common)} cells')
    paired = []
    for a, b in [('Emotional Pushback', 'Role Play'),
                 ('Emotional Pushback', 'Purpose Reverse'),
                 ('Role Play', 'Purpose Reverse')]:
        if a not in common or b not in common:
            continue


        diff = (common[b] - common[a]).rename('d').reset_index()
        point, low, high = estimate(diff, 'd')
        span = '' if low is None else f' [{low:+.3f}, {high:+.3f}]'
        print(f'  {a} minus {b}: {point:+.3f}{span}  paired n={len(common)}')
        paired.append({'first': a, 'second': b, 'drift_difference': point,
                       'low': low, 'high': high, 'paired_n': len(common)})
    pd.DataFrame(paired).to_csv(OUT / 'e2_methods.csv', index=False)
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['embed', 'analyse'], required=True)
    parser.add_argument('--primary-scenarios', type=int, default=50)
    parser.add_argument('--control-scenarios', type=int, default=50)
    parser.add_argument('--dialogues', type=int, default=0)
    parser.add_argument('--instrument', choices=sorted(INSTRUMENTS),
                        default='minilm')
    args = parser.parse_args()
    chosen = INSTRUMENTS[args.instrument]
    globals().update(INSTRUMENT=args.instrument, ENCODER=chosen['encoder'],
                     REVISION=chosen['revision'],
                     CACHE=ROOT / chosen['cache'], OUT=ROOT / chosen['cache'])
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f'instrument {args.instrument}: {chosen["encoder"]} at '
          f'{chosen["revision"][:12]}, cache {CACHE}')
    (stage_embed if args.stage == 'embed' else stage_analyse)(args)
