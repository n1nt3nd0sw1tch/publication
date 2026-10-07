"""Compute bootstrap intervals for annotation agreement.

Notes
-----
Resample the calibration data within model strata and update agreement bounds.
"""

import numpy as np
import pandas as pd

from settings import ANNOTATION_DIR, SAFETY, measure_column

DRAWS = 4000
SEED = 7
STRATUM = 'model'

FIELDS = ['answer'] + [measure_column(name) for name in SAFETY]
CONDITIONAL = 'alternative_response'


def kappa(human, judge):
    both = human.str.strip().ne('') & judge.str.strip().ne('')
    human, judge = human[both], judge[both]
    if len(human) == 0:
        return np.nan
    observed = (human == judge).mean()
    chance = sum((human == value).mean() * (judge == value).mean()
                 for value in set(human) | set(judge))
    return np.nan if chance >= 1 else (observed - chance) / (1 - chance)


def scored(pairs, field):
    if field != CONDITIONAL:
        return pairs
    return pairs[pairs['answer_human'].eq('Refusal')
                 | pairs['delivery_response_human'].eq('No')]


def interval(pairs, field, draws=DRAWS, seed=SEED):
    rows = scored(pairs, field)
    blocks = [group.index.to_numpy() for _, group in rows.groupby(STRATUM)]
    if not blocks or min(len(block) for block in blocks) < 2:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    human, judge = f'{field}_human', f'{field}_judge'
    values = np.empty(draws)
    for draw in range(draws):
        picked = np.concatenate([rng.choice(block, len(block), replace=True)
                                 for block in blocks])
        sample = rows.loc[picked]
        values[draw] = kappa(sample[human], sample[judge])
    usable = values[~np.isnan(values)]
    if usable.size == 0:
        return np.nan, np.nan, 0
    return (float(np.percentile(usable, 2.5)),
            float(np.percentile(usable, 97.5)), int(usable.size))


def main():
    pairs = pd.read_csv(ANNOTATION_DIR / 'comparison.csv', dtype=str,
                        keep_default_na=False)
    frozen = pd.read_csv(ANNOTATION_DIR / 'agreement.csv')

    rows = []
    for field in FIELDS:
        subset = scored(pairs, field)
        point = kappa(subset[f'{field}_human'], subset[f'{field}_judge'])
        low, high, usable = interval(pairs, field)
        rows.append({'field': field, 'recomputed': point, 'rows': len(subset),
                     'kappa_low': low, 'kappa_high': high, 'draws': usable})
    computed = pd.DataFrame(rows)

    check = frozen.merge(computed, on='field')
    difference = (check['kappa'] - check['recomputed']).abs()
    drift = check[(difference > 0.0005)
                  | (check['kappa'].isna() ^ check['recomputed'].isna())]
    if len(drift):
        raise SystemExit(
            'comparison.csv does not reproduce agreement.csv on '
            + ', '.join(drift['field']) + '. The saved pairs and the frozen '
            'coefficients are out of step, so no interval is written.')

    out = frozen.merge(computed[['field', 'kappa_low', 'kappa_high']], on='field')
    out.to_csv(ANNOTATION_DIR / 'agreement.csv', index=False)

    thin = computed[computed['draws'] < DRAWS]
    print(f'{len(computed)} coefficients checked against agreement.csv, '
          f'intervals written')
    for record in thin.itertuples():
        print(f'  {record.field}: {DRAWS - record.draws} of {DRAWS} resamples '
              f'gave no coefficient and were dropped')


if __name__ == '__main__':
    main()
