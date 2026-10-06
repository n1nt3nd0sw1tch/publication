"""Readability across the turns of Experiment 2.

A scratch script, not part of the pipeline. It lives at the repository root
rather than in scripts/, measures nothing that is written back, and can be
deleted without affecting a single reported number.

    python turns_readability.py            # measure, cache, report
    python turns_readability.py --remeasure # ignore the cache

Three measures, not fifteen. The grade level is the primary readability measure
of Section 3.4.3, Response Length is the complementary one that moves with it,
and Mean AoA is the vocabulary measure that separates a fall in word length from
a fall in sentence length. The other twelve are computed by scripts/language.py
for Experiment 1 and are not needed to see whether a model rewrites as a
dialogue is pressed.

Two tables for the chapter, levels and the paired change from the opening, and
two for the appendix behind it: the share of turns the fifty-word floor removes,
by model and turn, and the same change split by attack method.

The floor table is not decoration. The change columns rest on the dialogues long
enough at both ends, that share differs sharply across the panel and moves in
opposite directions within it, so the grade-level rows are not all measured on
comparably complete subsets. Without the table that caveat is an assertion.

Nothing here is confirmatory: Experiment 2 carries no declared readability
hypothesis, so every number below is descriptive.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'scripts' else HERE
sys.path.insert(0, str(ROOT / 'scripts'))

import evaluate                                              # noqa: E402
from analysis import NAME, ORDER, interval, macro_average    # noqa: E402
from language import FLOOR, measure_text                     # noqa: E402
from settings import TURNS_PATH, measure_column              # noqa: E402

CACHE = ROOT / 'results' / 'language' / 'turns_scratch.csv'

# The three measures, and whether each needs the fifty-word floor. Only the
# grade level does: Response Length is the length, and an age of acquisition is
# a mean over whatever words are there.
MEASURES = {
    'FKGL': True,
    'Response Length': False,
    'Mean AoA': False,
}

TURNS = [1, 2, 3]


def measured(remeasure=False):
    """Return one row per returned assistant turn, with the three measures."""
    if CACHE.exists() and not remeasure:
        return pd.read_csv(CACHE)

    turns = pd.read_csv(TURNS_PATH)
    turns = turns[turns['role'] == 'assistant'].copy()
    turns['turn'] = pd.to_numeric(turns['turn'], errors='coerce')
    turns = turns[turns['turn'].isin(TURNS)]

    # Returned turns only, on the rule of scripts/language.py: a turn the
    # provider withheld is empty, and measured it reports an intervention
    # outside the model as a model writing nothing.
    turns['text'] = turns['text'].fillna('').astype(str)
    withheld = (turns['text'].str.strip() == '').sum()
    turns = turns[turns['text'].str.strip() != '']

    norms = evaluate.load_aoa()
    rows = []
    for row in turns.itertuples():
        scored = measure_text(str(row.text), norms)
        rows.append({
            'dialogue_id': row.dialogue_id, 'scenario_id': row.scenario_id,
            'model': row.model, 'condition': row.condition,
            'method': row.method, 'turn': int(row.turn),
            **{name: scored[measure_column(name)] for name in MEASURES},
        })

    frame = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    print(f'Measured {len(frame):,} assistant turns, {withheld} withheld, '
          f'cached at {CACHE.relative_to(ROOT)}\n')
    return frame


def floored(frame, measure):
    """Blank the grade level on a turn too short for it to mean anything."""
    frame = frame.copy()
    frame[measure] = pd.to_numeric(frame[measure], errors='coerce')
    if MEASURES[measure]:
        length = pd.to_numeric(frame['Response Length'], errors='coerce')
        frame.loc[length < FLOOR, measure] = pd.NA
    return frame


def levels(frame):
    """Table 1. The macro-average of each measure at each turn."""
    print('Table 1 | Level by turn, macro-average over six models\n')
    print(f'  {"Measure":<18}' + ''.join(f'{f"Turn {t}":>26}' for t in TURNS))
    for measure in MEASURES:
        data = floored(frame, measure)
        cells = []
        for turn in TURNS:
            at_turn = data[data['turn'] == turn]
            per_model = {
                NAME.get(key, key):
                    part.groupby('scenario_id')[measure].mean()
                for key, part in at_turn.groupby('model')
            }
            cells.append(interval(*macro_average(per_model), places=2))
        print(f'  {measure:<18}' + ''.join(f'{c:>26}' for c in cells))
    print()


def paired(frame, measure, split='model'):
    """Per-scenario within-dialogue differences from turn 1, keyed by `split`.

    Paired within dialogue and resampled by scenario, as everywhere else. A
    dialogue enters a column only where both turns carry the measure, so the
    grade-level columns rest on the dialogues long enough at both ends.

    Splitting by model gives the panel rows. Splitting by method gives the
    appendix cut, and there the inner grouping stays the model, so a method
    figure is still a macro-average over six models and not a pooled count.
    """
    data = floored(frame, measure)
    keys = ['model', 'scenario_id', 'dialogue_id']
    if split not in keys:
        keys.insert(0, split)
    wide = data.pivot_table(index=keys, columns='turn', values=measure)

    effects = {}
    for later in (2, 3):
        if 1 not in wide.columns or later not in wide.columns:
            continue
        pair = wide[[1, later]].dropna()
        pair = (pair[later] - pair[1]).rename('diff').reset_index()
        pair['label'] = (pair['model'].map(NAME).fillna(pair['model'])
                         if split == 'model' else pair[split])
        effects[later] = {
            label: {
                NAME.get(inner, inner):
                    bit.groupby('scenario_id')['diff'].mean()
                for inner, bit in part.groupby('model')
            }
            for label, part in pair.groupby('label')
        }
    return effects


def report(effects, title, order, width=24, macro_row=True):
    """Print one change table: two turn columns, a scenario count, a macro row.

    macro_row is False for the method split. Averaging the three method figures
    is not the same arithmetic as averaging the six model figures: a scenario
    carries unequal numbers of dialogues per method once the floor has removed
    some, so equal weight on methods and equal weight on models give different
    numbers. On this corpus they differ by up to 0.06 grades, which is small
    enough to look like the panel figure and is not it. The panel macro-average
    is reported once, in Table 2, and the method rows are read against it.
    """
    print(f'{title}\n')
    print(f'  {"":<{width}}{"Turn 1 to 2":>22}{"Turn 1 to 3":>22}'
          f'{"Scenarios":>11}')
    for label in order:
        cells, count = [], 0
        for later in (2, 3):
            per_model = effects.get(later, {}).get(label)
            if not per_model:
                cells.append('---')
                continue
            cells.append(interval(*macro_average(per_model),
                                  places=2, sign=True))
            count = max(count, max(len(s) for s in per_model.values()))
        print(f'  {label:<{width}}{cells[0]:>22}{cells[1]:>22}{count:>11}')

    if not macro_row:
        print()
        return
    macro = []
    for later in (2, 3):
        per_label = effects.get(later, {})
        if not per_label:
            macro.append('---')
            continue
        # One series a model, taken from that model's own row.
        panel = {name: series
                 for label, block in per_label.items()
                 for name, series in block.items() if name == label}
        macro.append(interval(*macro_average(panel), places=2, sign=True))
    print(f'  {"Macro-average":<{width}}{macro[0]:>22}{macro[1]:>22}\n')


def change(frame, measure):
    """Table 2. The paired change from the opening turn, by model."""
    report(paired(frame, measure, 'model'),
           f'Table 2 | Change in {measure} from the opening turn', ORDER)


def by_method(frame, measure):
    """Appendix. The same change split by attack method.

    A method row is still a macro-average over six models, so it is comparable
    with the panel figure above it rather than being a pooled count.
    """
    effects = paired(frame, measure, 'method')
    methods = sorted({label for turn in effects.values() for label in turn})
    report(effects, f'Appendix | Change in {measure} by attack method',
           methods, macro_row=False)


def floor_table(frame):
    """Appendix. Share of turns below the floor, by model and turn.

    This is what the grade-level caveat rests on. The share is not constant
    across the panel and does not move the same way within it, so the
    grade-level rows are not all measured on comparably complete subsets.
    """
    length = pd.to_numeric(frame['Response Length'], errors='coerce')
    short = frame.assign(
        short=length < FLOOR,
        label=frame['model'].map(NAME).fillna(frame['model']))
    table = short.pivot_table(index='label', columns='turn', values='short')

    print('Appendix | Share of turns under the fifty-word floor (%)\n')
    print(f'  {"Model":<24}' + ''.join(f'{f"Turn {t}":>12}' for t in TURNS)
          + f'{"Change":>12}')
    for name in ORDER:
        if name not in table.index:
            continue
        row = [table.at[name, t] * 100 if t in table.columns else float('nan')
               for t in TURNS]
        print(f'  {name:<24}' + ''.join(f'{v:>11.1f}' for v in row)
              + f'{row[-1] - row[0]:>+12.1f}')
    panel = [short[short['turn'] == t]['short'].mean() * 100 for t in TURNS]
    print(f'  {"Panel":<24}' + ''.join(f'{v:>11.1f}' for v in panel)
          + f'{panel[-1] - panel[0]:>+12.1f}\n')
    print('  The floor removes most at the opening and least at turn 3, because\n'
          '  replies get longer as a dialogue is pressed. The change columns are\n'
          '  pairwise complete, which handles the pairing but not the selection.\n'
          '  Read the grade level against Response Length rather than alone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remeasure', action='store_true',
                        help='ignore the cache and measure again')
    arguments = parser.parse_args()

    frame = measured(arguments.remeasure)

    # Chapter.
    levels(frame)
    change(frame, 'FKGL')
    change(frame, 'Response Length')

    # Appendix.
    by_method(frame, 'FKGL')
    by_method(frame, 'Response Length')
    floor_table(frame)
