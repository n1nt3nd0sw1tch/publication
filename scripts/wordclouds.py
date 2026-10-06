"""Distinctive vocabulary by disclosure condition, as ten figures.

    python scripts/wordclouds.py
    python scripts/wordclouds.py --top 32 --minimum 8
    python scripts/wordclouds.py --only types

Writes into figures/:

    readability_words_type_{harmful,age_restricted,rights,benign}.pdf
    readability_words_model_{gpt,claude,gemini,deepseek,mistral,gemma}.pdf

brought in by figures/fig_readability_words.tex. Needs: pip install wordcloud

----------------------------------------------------------------------------
Two cuts of one analysis
----------------------------------------------------------------------------

The first four figures fix the scenario type and pool the models, and the last
six fix the model and pool the types. Between them they answer two questions the
same scoring cannot answer at once: whether the vocabulary moves with age
differently depending on what was asked, and whether it moves differently
depending on which model was asked.

Holding one of the two constant is not a presentational choice. The four
scenario types expect different answers, so a word distinctive to a harmful
scenario is distinctive because it belongs to a refusal and one distinctive to a
rights scenario because it belongs to an answer; a figure pooling both reports
the mixture. The same argument runs the other way for models whose reply lengths
differ by a factor of three.

----------------------------------------------------------------------------
Nine panels, and why the control is one of them
----------------------------------------------------------------------------

Three by three: the eight stated ages in order, then the control. The control
earns its panel because it is the baseline every age contrast in Chapter 4 is
read against, and a reader who cannot see what a model says when it is told
nothing has no reference for what it says when it is told an age. It sits last
rather than first because the grid is a progression and the control is not a
point on it: at the head it reads as an age below seven, and at the foot as the
reference the eight are measured against.

----------------------------------------------------------------------------
Each word is drawn once, at the condition where it scores highest
----------------------------------------------------------------------------

Scoring each condition independently put 'help', 'adult', 'parent' and 'school'
in every panel, because they are commoner at every minor age than at eighteen,
and the grid then read as one cloud repeated. So the scores are computed for all
nine conditions first and each word is kept only where its score is highest. The
assignment is the argument maximum of a score the corpus produced; nothing is
filtered by hand.

The cost is that a panel shows what separates it from its neighbours rather than
everything characteristic of it, which is the right trade for a figure whose
subject is the progression across the panels.

----------------------------------------------------------------------------
What these show, and what they must not be read as
----------------------------------------------------------------------------

Size is the weighted log-odds z-score from language.distinctive_words(): the log
odds of a word under one condition against the rest, with an informative
Dirichlet prior taken from the pooled cut, divided by its estimated standard
error. Dividing by the standard error is what makes it a z-score rather than a
ratio, and it is what stops a word seen a handful of times from outranking a
word seen a thousand. A word is large only where it is commoner in that
condition than the cut as a whole predicts, and it is not a frequency.
Position and orientation carry nothing, and each panel is scaled to its own
strongest word, so sizes rank within a panel and are not compared across them.

Tone carries the same ranking as size, on one perceptually uniform map read over
its dark half. Encoding the score twice is deliberate: relative area is hard to
judge in a packed layout and a reader can rank two words by tone when they
cannot by size.

Read as subject matter, never as vocabulary difficulty. Section 4.3.3 reports
mean AoA flat at 4.97 to 5.22 while grade level moves two to three grades, and
the two are consistent: a few dozen distinctive words out of a vocabulary of
thousands do not move a mean over matched tokens.

No floor is applied, unlike every other readability figure, since a word is a
word at any length and the fifty-word cut would remove the short refusals that
carry most of the referral vocabulary.
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

# The ladder, then the control. Read left to right and top to bottom, the first
# eight panels run from the youngest stated age to the oldest and the ninth is
# the neutral condition, which is what the model says when it is told nothing.
# The label is Neutral, matching config/settings.yml and every table in
# Chapters 3 and 4, so the figure and the text name one thing one way.
#
# The control sits last rather than first because the figure is a progression
# and the control is not a point on it. Placed at the head of the grid it reads
# as an age below seven; placed at the foot it reads as the reference the eight
# are measured against, which is what Chapter 4 uses it for.
LADDER = (7, 9, 11, 13, 15, 17, 18, 21)
CONDITIONS = ([(f'age{age:02d}', f'Age {age}') for age in LADDER]
              + [('neutral', 'Neutral')])

TYPES = ['Harmful', 'Age Restricted', 'Rights', 'Benign']

# One map for every figure, so a panel in one is read the same way as a panel in
# another, and read over almost its whole range so the grid is as legible by
# colour as by size.
#
# Tone still ranks: the most distinctive word in a panel is the dark violet at
# the foot of the map and the least is the yellow-green at its head. The bright
# end measures only about 1.3:1 against white, so those words are faint, and
# that is a deliberate consequence rather than an oversight. They are also the
# smallest words in the panel, and what the figure is for is the handful at the
# top of each; a range dark enough to make the twenty-eighth word crisp
# collapses the top eight into one shade and loses the ranking entirely.
COLOURMAP, TONE_DARK, TONE_PALE = colormaps['viridis'], 0.02, 0.88

# The panel outline, taken from the same map so the frame belongs to the palette
# rather than sitting outside it.
OUTLINE = colors.to_hex(COLOURMAP(0.72))

# The text block is 16cm wide: A4 at 21cm less the two 2.5cm margins that
# style/preamble.tex sets. Used to work out how far a figure is scaled down when
# it is included, which is what decides the panel label size.
TEXT_WIDTH_CM = 16.0

# The panel label size wanted in the finished document, in points. Matplotlib
# sets text in points of the figure it is drawn on, and a figure included at
# half the text width is scaled to about a third of its native size, so a label
# set at 7pt here arrives at about 2pt on the page. The size is therefore
# computed from the width the figure will be shown at rather than fixed, and it
# is large in the raw PDF, which is never the thing a reader sees.
#
# Seven, not nine. The label is set inside the figure and competes with the
# panels for the same vertical space, so a target that reads comfortably as body
# text pushes the rows apart and, at nine, ran the label of one row into the
# panel above it. Seven is a shade under the footnotesize the table captions use
# and is the largest that leaves the grid its room.
LABEL_POINTS = 7.0

# The clouds and their labels take matplotlib's default face. A serif was tried
# to match the document and reverted: a word cloud sets a word at whatever size
# its score earns, and at the small end a serif loses more legibility than the
# match is worth.

SIGNPOST = {'parent', 'parents', 'guardian', 'guardians', 'teacher', 'teachers',
            'counselor', 'counsellor', 'adult', 'adults', 'trusted',
            'caregiver', 'caregivers', 'school', 'family', 'someone',
            'mom', 'dad', 'mum', 'grandparent', 'nurse', 'coach'}


# Define function to read the replies, keeping the control alongside the ages
def load_conditions():
    replies = language.load_texts()
    # The four cue conditions are dropped here. They are the weaker form of
    # disclosure and carry no stated age, so they have no place on a ladder
    # whose panels are ordered by one.
    wanted = {key for key, _ in CONDITIONS}
    replies = replies[replies['condition'].isin(wanted)].copy()
    return replies.assign(key=replies['condition'])


# Define function to score every condition within one cut, then keep each word
# only where it peaks
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


# Define function to draw one panel, tone tracking the rank within it
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
            # darkest for the most distinctive word in this panel, palest for
            # the least, so tone and size rank the words the same way
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


# Define function to draw one cut as a three by three grid
#
# No overall title. The caption names the cut, and a heading inside the image
# would say it twice at a size nothing else in the document uses.
def draw_grid(assigned, top, filename, display):
    width = 9.9
    figure, axes = plt.subplots(3, 3, figsize=(width, 7.9))

    # How far the figure shrinks when \includegraphics sets it to a fraction of
    # the text block, and the label size that lands at LABEL_POINTS after it.
    scale = (display * TEXT_WIDTH_CM) / (width * 2.54)
    points = LABEL_POINTS / scale

    for index, (key, label) in enumerate(CONDITIONS):
        axis = axes[index // 3][index % 3]
        draw(assigned[key], axis, top)
        axis.set_title(label, fontsize=points, color=INK,
                       pad=points * 0.22)

    # tight_layout measures its padding in multiples of the default font size,
    # which is ten points and has nothing to do with the label size computed
    # above. Left alone it reserves a tenth of what a label this size needs and
    # the rows collide. Scaling it by the label keeps the gap proportionate at
    # any display width.
    figure.tight_layout(h_pad=points * 0.14, w_pad=0.7)

    written = FIGURES / filename
    figure.savefig(written, bbox_inches='tight')
    plt.close(figure)
    return written


# Define function to report what a grid holds, since a cloud is unreadable as
# evidence and the words behind it are what a claim rests on
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

    # Each figure is drawn inside its own guard. Ten grids over a corpus this
    # size take a few minutes, and a failure on the third should not cost the
    # seven that would have followed it: the run reports which cut failed and
    # carries on, so one bad cut is a missing file and a printed reason rather
    # than a silent short set.
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
