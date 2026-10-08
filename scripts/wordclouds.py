"""Generate age-conditioned distinctive-vocabulary word clouds.

Notes
-----
Compare eight explicit ages and the Neutral condition across all scenarios.
Normalize plural forms before ranking words by weighted log-odds.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import MUTED
from settings import ROOT


FIGURES = ROOT / "figures" / "readability"
AGES = (7, 9, 11, 13, 15, 17, 18, 21)
CONDITIONS = [(f"age{age:02d}", f"Age {age}") for age in AGES] + [("neutral", "Neutral")]

PALETTE = (
    "#0B4F8C",
    "#0B775E",
    "#A23B72",
    "#B86E00",
    "#2C7FB8",
    "#A84300",
    "#B79F00",
)
OUTLINE = "#008F5A"
INK = "#202326"
BACKGROUND = "#FFFFFF"

FIGURE_SIZE = (11.2, 8.2)
TITLE_SIZE = 16
RESOLUTION = 240

# Only clean tokens for this word cloud; never alter the source response texts
# or the readability measurements. Keep these choices explicit and auditable.
TOKEN = re.compile(r"[a-z]+")
ARTIFACTS = {
    "http", "https", "www", "com", "org", "html", "href", "mailto",
    "nbsp", "amp", "utm", "jpeg", "png", "pdf", "markdown",
}

# Standard English irregular plurals and selected obvious inflections.
IRREGULAR = {
    "children": "child",
    "people": "person",
    "women": "woman",
    "men": "man",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "crises": "crisis",
    "analyses": "analysis",
    "diagnoses": "diagnosis",
    "counsellors": "counselor",
    "counsellor": "counselor",
    "counselors": "counselor",
    "movies": "movie",
    "cookies": "cookie",
    "brownies": "brownie",
    "smoothies": "smoothie",
    "selfies": "selfie",
    "zombies": "zombie",
    "rookies": "rookie",
    "ties": "tie",
    "pies": "pie",
}

# Do not strip the final "s" from these singular nouns or non-plural words.
INVARIANT = {
    "crisis", "analysis", "diagnosis", "thesis", "status", "loss",
    "stress", "class", "process", "access", "success", "address",
    "news", "series", "species", "physics", "mathematics", "diabetes",
    "business", "glass", "grass", "discuss", "across", "always",
    "perhaps", "plus", "virus", "campus", "bonus", "bias", "gas",
    "focus", "cannabis", "serious", "curious", "various",
}


@lru_cache(maxsize=20000)
def canonical_word(word: str) -> str:
    """Conservatively merge common plural forms without aggressive stemming.

    Notes
    -----
    Preserve ordinary lexical meaning: ``parents`` -> ``parent`` and
    ``families`` -> ``family``, but ``parental`` remains distinct. This
    intentionally does not attempt context-dependent verb lemmatization.
    """
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word in INVARIANT or len(word) < 4:
        return word
    if word.endswith("ies") and len(word) > 4:
        # "families" -> "family"; common -ie exceptions are listed above.
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes", "zzes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(
        ("ss", "us", "is", "ous", "ics", "as")
    ):
        return word[:-1]
    return word


def clean_reply(text: str) -> str:
    """Remove response formatting and normalize tokens before scoring.

    Notes
    -----
    Use the same stopword list as the original weighted log-odds analysis.
    Remove stopwords both before and after inflection merging, preventing
    generic words from resurfacing under a newly normalized form.
    """
    plain = language.clean(text).lower()
    words = []
    for token in TOKEN.findall(plain):
        if len(token) < 3 or token in language.STOPWORDS or token in ARTIFACTS:
            continue
        base = canonical_word(token)
        if len(base) >= 3 and base not in language.STOPWORDS and base not in ARTIFACTS:
            words.append(base)
    return " ".join(words)


def load_conditions():
    """Load all age-conditioned and Neutral replies."""
    replies = language.load_texts()
    wanted = {key for key, _ in CONDITIONS}
    replies = replies.loc[replies["condition"].isin(wanted)].copy()
    replies["key"] = replies["condition"]
    replies["cloud_text"] = replies["response"].map(clean_reply)
    return replies


def assign_words(replies, minimum):
    """Assign each distinctive word to its strongest age contrast."""
    scores = {}

    for key, _ in CONDITIONS:
        here = replies.loc[replies["key"].eq(key), "cloud_text"]
        other = replies.loc[replies["key"].ne(key), "cloud_text"]
        values = language.distinctive_words(here, other, minimum=minimum)
        scores[key] = values[values > 0]

    assigned = {key: {} for key, _ in CONDITIONS}
    strongest = {}

    for key, values in scores.items():
        for word, score in values.items():
            if word not in strongest or score > strongest[word][1]:
                strongest[word] = (key, score)

    for word, (key, score) in strongest.items():
        assigned[key][word] = score

    return {
        key: dict(sorted(words.items(), key=lambda pair: -pair[1]))
        for key, words in assigned.items()
    }


def draw_cloud(ax, words, top, width, height):
    """Render a proportional word cloud inside one panel."""
    from wordcloud import WordCloud

    entries = dict(list(words.items())[:top])
    ax.set_facecolor(BACKGROUND)

    if entries:
        ranks = {word: rank for rank, word in enumerate(entries)}

        def color_func(word, **_):
            return PALETTE[ranks[word] % len(PALETTE)]

        cloud = WordCloud(
            width=width,
            height=height,
            background_color="white",
            max_words=top,
            prefer_horizontal=0.91,
            relative_scaling=0.48,
            min_font_size=9,
            max_font_size=130,
            margin=3,
            collocations=False,
            random_state=7,
            color_func=color_func,
        ).generate_from_frequencies(entries)
        ax.imshow(cloud, interpolation="bilinear", aspect="auto")
    else:
        ax.text(
            0.5,
            0.5,
            "No Distinctive Words",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color=MUTED,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(OUTLINE)
        spine.set_linewidth(1.05)


def draw_grid(assigned, top, filename):
    """Render a three-by-three grid without title overlaps."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        3, 3, figsize=FIGURE_SIZE, facecolor=BACKGROUND
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.948,
        bottom=0.035,
        wspace=0.055,
        hspace=0.27,
    )

    for ax, (key, label) in zip(axes.flat, CONDITIONS):
        box = ax.get_position()
        image_width = max(1, round(box.width * FIGURE_SIZE[0] * RESOLUTION))
        image_height = max(1, round(box.height * FIGURE_SIZE[1] * RESOLUTION))
        draw_cloud(ax, assigned.get(key, {}), top, image_width, image_height)
        ax.set_title(
            label,
            fontsize=TITLE_SIZE,
            color=INK,
            weight="normal",
            pad=5,
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    destination = FIGURES / filename
    fig.savefig(destination, dpi=300, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Figure: {destination.name}")
    return destination


def main(args):
    """Write the single age-conditioned vocabulary figure."""
    replies = load_conditions()
    if replies.empty:
        raise ValueError("No age-conditioned replies found")

    assigned = assign_words(replies, args.minimum)
    draw_grid(assigned, args.top, "readability_words_age.pdf")


def parser():
    """Parse figure-generation options."""
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--top", type=int, default=20)
    cli.add_argument("--minimum", type=int, default=10)
    return cli


if __name__ == "__main__":
    main(parser().parse_args())
