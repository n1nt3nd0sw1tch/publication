"""Generate age-based distinctive-vocabulary word clouds.

Notes
-----
Compare explicit age conditions and the neutral condition across the full
corpus. Words are assigned to the age where they are most distinctive.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import INK, MUTED
from settings import ROOT

FIGURES = ROOT / 'figures' / 'readability'

LADDER = (7, 9, 11, 13, 15, 17, 18, 21)
CONDITIONS = ([(f'age{age:02d}', f'Age {age}') for age in LADDER]
              + [('neutral', 'Neutral')])

# Colorblind-safe Okabe-Ito palette, used cyclically by rank.
PALETTE = [
    '#0072B2',  # blue
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#D55E00',  # vermilion
    '#F0E442',  # yellow
]
OUTLINE = '#00A651'
BACKGROUND = 'white'

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 7.0
TITLE_POINTS = 9.2

SIGNPOST = {
    'parent', 'parents', 'guardian', 'guardians', 'teacher', 'teachers',
    'counselor', 'counsellor', 'adult', 'adults', 'trusted',
    'caregiver', 'caregivers', 'school', 'family', 'someone',
    'mom', 'dad', 'mum', 'grandparent', 'nurse', 'coach'
}


def load_conditions():
    """Load replies for the explicit-age and neutral conditions."""
    replies = language.load_texts()
    wanted = {key for key, _ in CONDITIONS}
    replies = replies[replies['condition'].isin(wanted)].copy()
    return replies.assign(key=replies['condition'])



def assign_words(part, minimum):
    """Assign each word to the condition where it is most distinctive."""
    scored = {}
    for key, _ in CONDITIONS:
        here = part[part['key'] == key]['response']
        rest = part[part['key'] != key]['response']
        series = language.distinctive_words(here, rest, minimum=minimum)
        scored[key] = series[series > 0]

    best = {}
    for key, series in scored.items():
        for word, value in series.items():
            if word not in best or value > best[word][1]:
                best[word] = (key, value)

    assigned = {key: {} for key, _ in CONDITIONS}
    for word, (key, value) in best.items():
        assigned[key][word] = value

    return {
        key: dict(sorted(words.items(), key=lambda item: -item[1]))
        for key, words in assigned.items()
    }



def draw(words, axis, top):
    """Draw one word-cloud panel."""
    from wordcloud import WordCloud

    words = dict(list(words.items())[:top])
    if not words:
        axis.text(
            0.5,
            0.5,
            'no distinctive words',
            ha='center',
            va='center',
            fontsize=8,
            color=MUTED,
        )
    else:
        ranks = {word: index for index, word in enumerate(words)}

        def tone(word, **kwargs):
            return PALETTE[ranks[word] % len(PALETTE)]

        cloud = WordCloud(
            width=680,
            height=500,
            background_color=BACKGROUND,
            prefer_horizontal=0.90,
            relative_scaling=0.52,
            min_font_size=6,
            max_words=top,
            color_func=tone,
            random_state=7,
            collocations=False,
            margin=4,
        ).generate_from_frequencies(words)
        axis.imshow(cloud, interpolation='bilinear')

    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor(OUTLINE)
        spine.set_linewidth(1.6)



def draw_grid(assigned, top, filename, display):
    """Draw the 3x3 age-condition grid."""
    width = 10.0
    height = 7.4
    figure, axes = plt.subplots(3, 3, figsize=(width, height), facecolor=BACKGROUND)

    scale = (display * TEXT_WIDTH_CM) / (width * 2.54)
    points = LABEL_POINTS / scale
    title_points = TITLE_POINTS / scale

    for index, (key, label) in enumerate(CONDITIONS):
        axis = axes[index // 3][index % 3]
        draw(assigned[key], axis, top)
        axis.set_title(label, fontsize=points, color=INK, pad=points * 0.28)

    figure.suptitle(
        'Unique Word Clouds by Age Condition',
        fontsize=title_points,
        color=INK,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=points * 0.18, w_pad=0.75)

    written = FIGURES / filename
    figure.savefig(written, bbox_inches='tight', facecolor=BACKGROUND)
    plt.close(figure)
    return written



def report(title, assigned, written, top):
    """Print a compact report of the exported figure."""
    print(f'{title}   {written.name}')
    for key, label in CONDITIONS:
        words = list(assigned[key])[:top]
        named = sum(1 for word in words if word in SIGNPOST)
        print(f'  {label:<8} {len(assigned[key]):>4} words  '
              f'{named}/{len(words)}  {", ".join(words)}')
    print()



def main(arguments):
    """Generate age-based distinctive-vocabulary word clouds."""
    replies = load_conditions()
    stated = int((replies['key'] != 'neutral').sum())
    control = len(replies) - stated
    print(f'{len(replies):,} replies across {len(CONDITIONS)} conditions: '
          f'{stated:,} at a stated age and {control:,} under the control\n')
    FIGURES.mkdir(exist_ok=True)

    try:
        assigned = assign_words(replies, arguments.minimum)
        written = draw_grid(
            assigned,
            arguments.top,
            'readability_words_age.pdf',
            arguments.display,
        )
    except Exception as failure:
        print(f'Age   FAILED, {type(failure).__name__}: {failure}\n')
        return

    report('Age', assigned, written, arguments.top)
    print(f'Written to {FIGURES.relative_to(ROOT)}')



def parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--top',
        type=int,
        default=15,
        help='words drawn in each panel, after assignment',
    )
    parser.add_argument(
        '--minimum',
        type=int,
        default=10,
        help='times a word must appear in the cut before it can be scored',
    )
    parser.add_argument(
        '--display',
        type=float,
        default=0.48,
        help='fraction of the text width the figure will be included at',
    )
    return parser


if __name__ == '__main__':
    main(parser().parse_args())
