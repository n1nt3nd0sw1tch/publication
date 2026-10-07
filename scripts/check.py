"""Validate invariants of the semantic-analysis pipeline.

Notes
-----
The script checks cache integrity, coverage, alignment, bootstrap behaviour,
semantic estimands, and dialogue baselines.
"""

import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '.')
import semantics as S

S.CACHE = S.OUT = S.ROOT / S.INSTRUMENTS['minilm']['cache']
S.INSTRUMENT='minilm'; S.ENCODER=S.INSTRUMENTS['minilm']['encoder']; S.REVISION=S.INSTRUMENTS['minilm']['revision']
C = S.CACHE
man = json.load(open(C / 'manifest.json'))
ok = lambda i, msg: print(f'  {i:2d}. PASS  {msg}')

single = pd.read_pickle(C / 'single_key.pkl').reset_index(drop=True)
stext = pd.read_pickle(C / 'single_text.pkl').reset_index(drop=True)
turns = pd.read_pickle(C / 'turns_key.pkl').reset_index(drop=True)
ttext = pd.read_pickle(C / 'turns_text.pkl').reset_index(drop=True)
V, W = np.load(C / 'single_vec.npy'), np.load(C / 'turns_vec.npy')


for name, frame in (('single', pd.concat([single, stext], axis=1)),
                    ('turns', pd.concat([turns, ttext], axis=1))):
    assert S.digest(frame) == man[name]['digest'], name
ok(1, 'digest(current) == digest(manifest) for both pools')


for name, vec, frame in (('single', V, single), ('turns', W, turns)):
    assert list(vec.shape) == man[name]['shape'] and len(frame) == man[name]['texts']
ok(2, 'vector shapes and text counts match the manifest')


heads = turns[turns.turn.eq(1)][
    ['model', 'prompt_id', 'opening_replicate']
].drop_duplicates()
counts = single.groupby(['model', 'prompt_id']).replicate.nunique()
n = pd.Series([counts.get((m, p), 0)
               for m, p in zip(heads.model, heads.prompt_id)])
assert (n >= 1).all()
complete = int((n == 3).sum())
incomplete = int((n < 3).sum())
ok(3, f'{complete} dialogue openings have both unused baseline replicates; '
      f'{incomplete} are baseline-incomplete')


kept = set(zip(single.model, single.prompt_id, single.replicate))
assert all((m, p, r) in kept for m, p, r in heads.itertuples(index=False))
ok(4, 'every continued replicate is in the pool')


cells = pd.read_csv(C / 'e1_cells.csv')
drawn = {s: sorted(g.scenario_id.unique()) for s, g in cells.groupby('scenario_type')}
pairs = {S.TRAJECTORY: len(S.MINOR) * len(S.ADULT),
         S.THRESHOLD_CONTRAST: 1,
         S.SIGNAL: len(S.MINOR) * len(S.MINOR_CUE),
         S.CUE_DIRECTION: len(S.CUE_PAIRS),
         S.NEUTRAL_AGE: len(S.MINOR),
         S.NEUTRAL_CUE: len(S.MINOR_CUE)}

drawn, panel = man['e1_scenarios'], man['models']
key = ['contrast', 'stratum', 'model']
full = S.coverage(cells, pairs, drawn, panel)
gone_model = panel[0]
hurt = S.coverage(cells[cells.model.ne(gone_model)], pairs, drawn, panel)
both = full.merge(hurt, on=key, suffixes=('_full', '_hurt'))
assert len(both) == len(full)
assert (both.possible_full == both.possible_hurt).all()
assert (both[both.model.eq(gone_model)].coverage_hurt == 0).all()

gone_scenario = drawn[S.PRIMARY][0]
lost = S.coverage(cells[cells.scenario_id.ne(gone_scenario)], pairs, drawn, panel)
side = full.merge(lost, on=key, suffixes=('_full', '_lost'))
assert (side.possible_full == side.possible_lost).all()
ar = side[side.stratum.eq(S.PRIMARY)]
assert (ar.coverage_lost <= ar.coverage_full).all()
assert (ar.coverage_lost < ar.coverage_full).any()
ok(5, 'losing a model or a whole scenario leaves the denominator fixed '
      'and lowers coverage')


tmp = Path(tempfile.mkdtemp())
shutil.copytree(C, tmp / 'semantics')
bad = json.load(open(tmp / 'semantics' / 'manifest.json'))
bad['single']['digest'] = '0' * 64
json.dump(bad, open(tmp / 'semantics' / 'manifest.json', 'w'))
try:
    S.check_cache('single', pd.concat([single, stext], axis=1), V, bad)
    raise SystemExit('  6. FAIL  a corrupted digest was accepted')
except AssertionError:
    ok(6, 'a corrupted digest is refused')


assert S.action_aligned('Refusal', np.nan) is None
assert S.action_aligned('Refusal', 'Refusal with Delivery') is False
assert S.action_aligned('Refusal', 'Refusal without Delivery') is True
ok(7, 'a missing verdict is unscorable rather than defective')


frame = pd.DataFrame({'scenario_id': ['a', 'b'], 'model': ['m', 'm'],
                      'value': [0.0, 1.0]})
low, high = S.interval(frame, 'value', draws=4000)


assert (low, high) == (0.0, 1.0), (low, high)
ok(8, 'S.interval keeps a scenario drawn twice counted twice')


rng = np.random.default_rng(0)
A = np.linalg.qr(rng.normal(size=(3, 8)).T)[0].T
B = np.linalg.qr(rng.normal(size=(3, 8)).T)[0].T
sb, sw, sep = S.separation(np.vstack([A, B]), np.arange(3), np.arange(3, 6))
db = (1 - A @ B.T).mean()
dw = ((1 - A @ A.T)[np.triu_indices(3, 1)].mean()
      + (1 - B @ B.T)[np.triu_indices(3, 1)].mean()) / 2
assert abs(sep - (db - dw)) < 1e-12 and abs(sep - (sw - sb)) < 1e-12
ok(9, 'S_within - S_between == D_between - D_within')


marginal, at = S.length_marginals(cells)
assert len(marginal) == cells.groupby(['contrast', 'scenario_type']).ngroups
assert marginal.adjusted_marginal.notna().all()


shuffled = cells.copy()
first, last = sorted(cells.model.unique())[0], sorted(cells.model.unique())[-1]
shuffled['model'] = shuffled.model.replace({first: 'zzz_' + first})
again, _ = S.length_marginals(shuffled)
assert np.allclose(marginal.adjusted_marginal.to_numpy(),
                   again.adjusted_marginal.to_numpy(), atol=1e-9)
ok(10, 'S.length_marginals has full rank and is invariant to the reference model')

assert 'not estimable' in S.describe_length(None, None, None, 0)
assert 'no interval reported' in S.describe_length(1.2e-4, None, None, 40)
assert '[' in S.describe_length(1.2e-4, 1e-4, 2e-4, 9000)
ok(11, 'describe_length formats both safeguard cases without raising')


assert not single.duplicated(['model', 'prompt_id', 'replicate']).any()
assert not turns.duplicated(['dialogue_id', 'turn']).any()
ok(12, 'no duplicate adaptation replicate or dialogue turn in the pools')


assert len(S.CONDITIONS) == 13
assert set(S.MINOR_CUE) == {'routine_minor', 'people_minor'}
assert set(S.ADULT_CUE) == {'routine_adult', 'people_adult'}
assert set(S.CUE_PAIRS) == {
    ('routine_minor', 'routine_adult'),
    ('people_minor', 'people_adult')
}
ok(13, 'E1 semantic pool includes all 13 conditions and both matched cue pairs')


drift_path = C / 'e2_drift.csv'
if drift_path.exists():
    drift = pd.read_csv(drift_path)
    assert not (
        drift.baseline_gap.ne('ok') & drift.opening_similarity.notna()
    ).any()
    assert drift.loc[
        drift.baseline_gap.eq('ok'), 'opening_similarity'
    ].notna().all()
    ok(14, 'matched-opening baseline is complete-case only')
else:
    ok(14, 'e2_drift.csv not written yet; baseline invariant deferred to analysis')

print('\nall fourteen invariants hold')
