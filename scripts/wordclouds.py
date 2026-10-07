"""Generate distinctive-vocabulary word-cloud figures.

Notes
-----
Compare disclosure conditions and export scenario- and model-level word clouds.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps, colors

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import INK, MUTED, NAME, ORDER, PALE
from settings import ROOT

FIGURES = ROOT / 'figures'


LADDER = (7, 9, 11, 13, 15, 17, 18, 21)
CONDITIONS = ([(f'age{age:02d}', f'Age {age}') for age in LADDER]
              + [('neutral', 'Neutral')])

TYPES = ['Harmful', 'Age Restricted', 'Rights', 'Benign']


COLOURMAP, TONE_DARK, TONE_PALE = colormaps['viridis'], 0.02, 0.88


OUTLINE = colors.to_hex(COLOURMAP(0.72))


TEXT_WIDTH_CM = 16.0


LABEL_POINTS = 7.0


SIGNPOST = {'parent', 'parents', 'guardian', 'guardians', 'teacher', 'teachers',
            'counselor', 'counsellor', 'adult', 'adults', 'trusted',
            'caregiver', 'caregivers', 'school', 'family', 'someone',
            'mom', 'dad', 'mum', 'grandparent', 'nurse', 'coach'}


def load_conditions():
    replies = language.load_texts()


    wanted = {key for key, _ in CONDITIONS}
    replies = replies[replies['condition'].isin(wanted)].copy()
    return replies.assign(key=replies['condition'])


def assign_words(part, minimum):
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
    return {key: dict(sorted(words.items(), key=lambda item: -item[1]))
            for key, words in assigned.items()}


def draw(words, axis, top):
    from wordcloud import WordCloud

    words = dict(list(words.items())[:top])
    if not words:
        axis.text(0.5, 0.5, 'no words peak here', ha='center', va='center',
                  fontsize=8, color=MUTED)
    else:
        ranks = {word: index for index, word in enumerate(words)}
        span = max(len(words) - 1, 1)

        def tone(word, **kwargs):


            position = ranks[word] / span
            return colors.to_hex(
                COLOURMAP(TONE_DARK + position * (TONE_PALE - TONE_DARK)))

        cloud = WordCloud(width=680, height=500, background_color='white',
                          prefer_horizontal=0.88,
                          relative_scaling=0.55, min_font_size=6,
                          max_words=top, color_func=tone,
                          random_state=7).generate_from_frequencies(words)
        axis.imshow(cloud, interpolation='bilinear')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor(OUTLINE)
        spine.set_linewidth(1.1)


def draw_grid(assigned, top, filename, display):
    width = 9.9
    figure, axes = plt.subplots(3, 3, figsize=(width, 7.9))


    scale = (display * TEXT_WIDTH_CM) / (width * 2.54)
    points = LABEL_POINTS / scale

    for index, (key, label) in enumerate(CONDITIONS):
        axis = axes[index // 3][index % 3]
        draw(assigned[key], axis, top)
        axis.set_title(label, fontsize=points, color=INK,
                       pad=points * 0.22)


    figure.tight_layout(h_pad=points * 0.14, w_pad=0.7)

    written = FIGURES / filename
    figure.savefig(written, bbox_inches='tight')
    plt.close(figure)
    return written


def report(title, assigned, written, top):
    print(f'{title}   {written.name}')
    for key, label in CONDITIONS:
        words = list(assigned[key])[:top]
        named = sum(1 for word in words if word in SIGNPOST)
        print(f'  {label:<8} {len(assigned[key]):>4} words  '
              f'{named}/{len(words)}  {", ".join(words)}')
    print()


def main(arguments):
    replies = load_conditions()
    stated = int((replies['key'] != 'neutral').sum())
    control = len(replies) - stated
    print(f'{len(replies):,} replies across {len(CONDITIONS)} conditions: '
          f'{stated:,} at a stated age and {control:,} under the control\n')
    FIGURES.mkdir(exist_ok=True)


    def build(title, part, filename):
        if part.empty:
            print(f'{title}   no replies, skipped\n')
            return
        try:
            assigned = assign_words(part, arguments.minimum)
            written = draw_grid(assigned, arguments.top, filename,
                                arguments.display)
        except Exception as failure:
            print(f'{title}   FAILED, {type(failure).__name__}: {failure}\n')
            return
        report(title, assigned, written, arguments.top)

    if arguments.only in ('types', 'both'):
        present = set(replies['scenario_type'].unique())
        missing = [kind for kind in TYPES if kind not in present]
        if missing:
            print(f'scenario types absent from the corpus: '
                  f'{", ".join(missing)}')
            print(f'present: {", ".join(sorted(present))}\n')
        for kind in TYPES:
            build(kind, replies[replies['scenario_type'] == kind],
                  f'readability_words_type_'
                  f'{kind.lower().replace(" ", "_")}.pdf')

    if arguments.only in ('models', 'both'):
        slugs = {'GPT-5.6 Luna': 'gpt', 'Claude Haiku 4.5': 'claude',
                 'Gemini 3.5 Flash Lite': 'gemini',
                 'DeepSeek-V4 Flash': 'deepseek', 'Mistral Small 4': 'mistral',
                 'Gemma 4 31B': 'gemma'}
        replies['label'] = replies['model'].map(NAME)
        for label in ORDER:
            build(label, replies[replies['label'] == label],
                  f'readability_words_model_{slugs[label]}.pdf')

    print(f'Written to {FIGURES.relative_to(ROOT)}')
    print('Upload them to Overleaf: figures/fig_readability_words.tex expects '
          'them there, and the Overleaf tooling writes text only.')


def parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--top', type=int, default=10,
                        help='words drawn in each panel, after assignment')
    parser.add_argument('--minimum', type=int, default=10,
                        help='times a word must appear in the cut before it '
                             'can be scored')
    parser.add_argument('--display', type=float, default=0.48,
                        help='fraction of the text width the figure will be '
                             'included at, which sets the panel label size')
    parser.add_argument('--only', default='both',
                        choices=['both', 'types', 'models'],
                        help='draw one set of figures rather than both')
    return parser


if __name__ == '__main__':
    main(parser().parse_args())
