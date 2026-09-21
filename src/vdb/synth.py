"""Mini-CLEVR: deterministic, procedurally generated visual QA.

Everything is generated from a seed — images, questions, and labels — so a
`vdb bench` run is byte-reproducible on any machine and needs no dataset
downloads or license headaches. Task families: counting, color, shape, and
spatial relation; answers are categorical, which makes logit-level
distillation (KL over the answer simplex) meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPES = ("circle", "square")
COLORS = ("red", "green", "blue", "yellow")
SIZES = ("small", "large")
COLOR_RGB = {
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.75, 0.25),
    "blue": (0.2, 0.35, 0.9),
    "yellow": (0.95, 0.85, 0.2),
}

#: answer vocabulary shared by every task family
ANSWERS = tuple(
    [str(i) for i in range(4)] + list(COLORS) + list(SHAPES) + ["yes", "no"]
)
ANSWER_TO_ID = {a: i for i, a in enumerate(ANSWERS)}

TOKENS = tuple(
    ["[PAD]", "how", "many", "what", "shape", "color", "is", "the", "largest",
     "smallest", "left", "object", "objects", "first", "second"]
    + [s + "s" for s in SHAPES]
    + list(SHAPES)
)
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}

IMG = 32
_CELL = IMG // 4  # 4x4 placement grid


@dataclass(frozen=True)
class Example:
    image: np.ndarray  # float32 [3, IMG, IMG] in [0, 1]
    question: tuple[int, ...]
    answer: int
    family: str


def _render(rng: np.random.Generator) -> list[dict]:
    """Render 1-3 non-overlapping objects on a 4x4 grid."""
    n = int(rng.integers(1, 4))
    objects = []
    for cell in rng.permutation(16)[:n]:
        row, col = divmod(int(cell), 4)
        size = str(rng.choice(SIZES))
        objects.append({
            "shape": str(rng.choice(SHAPES)),
            "color": str(rng.choice(COLORS)),
            "size": size,
            "x": (col + 0.5) * _CELL,
            "y": (row + 0.5) * _CELL,
            "r": _CELL * (0.22 if size == "small" else 0.42),
            "col": col,
        })
    return objects


def _paint(objects: list[dict]) -> np.ndarray:
    img = np.zeros((3, IMG, IMG), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(IMG, dtype=np.float32),
                         np.arange(IMG, dtype=np.float32), indexing="ij")
    for o in objects:
        r, col = o["r"], COLOR_RGB[o["color"]]
        if o["shape"] == "circle":
            mask = (xx - o["x"]) ** 2 + (yy - o["y"]) ** 2 <= r * r
        else:
            mask = (np.abs(xx - o["x"]) <= r) & (np.abs(yy - o["y"]) <= r)
        for ch in range(3):
            img[ch][mask] = col[ch]
    return img


def _q(words: list[str]) -> tuple[int, ...]:
    return tuple(TOKEN_TO_ID[w] for w in words)


def _questions(objects: list[dict], rng: np.random.Generator) -> list[tuple[tuple[int, ...], int, str]]:
    target = objects[int(rng.integers(len(objects)))]
    qs: list[tuple[tuple[int, ...], int, str]] = [
        (_q(["how", "many", "objects"]), ANSWER_TO_ID[str(len(objects))], "count"),
        (_q(["what", "shape", "the", "object"]), ANSWER_TO_ID[target["shape"]], "shape"),
        (_q(["what", "color", "the", "largest", "object"]),
         ANSWER_TO_ID[max(objects, key=lambda o: o["r"])["color"]], "color"),
        (_q(["what", "color", "the", "smallest", "object"]),
         ANSWER_TO_ID[min(objects, key=lambda o: o["r"])["color"]], "color"),
    ]
    if len(objects) >= 2:
        a, b = objects[0], objects[1]
        answer = "yes" if a["col"] < b["col"] else "no"
        qs.append((_q(["is", "the", "first", "object", "left", "second"]),
                   ANSWER_TO_ID[answer], "spatial"))
    return qs


def generate(n_examples: int, seed: int = 0) -> list[Example]:
    """Produce `n_examples` (scene, question, answer) triples; deterministic."""
    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    while len(examples) < n_examples:
        objects = _render(rng)
        img = _paint(objects)
        for q, a, fam in _questions(objects, rng):
            examples.append(Example(image=img, question=q, answer=a, family=fam))
            if len(examples) >= n_examples:
                break
    return examples


def split(n_train: int, n_val: int, seed: int = 0) -> tuple[list[Example], list[Example]]:
    return generate(n_train, seed), generate(n_val, seed + 10_000)


def num_classes() -> int:
    return len(ANSWERS)


def vocab_size() -> int:
    return len(TOKENS)


def pad_tokens(examples: list[Example], max_len: int = 6) -> np.ndarray:
    q = np.zeros((len(examples), max_len), dtype=np.int64)
    for i, ex in enumerate(examples):
        toks = ex.question[:max_len]
        q[i, : len(toks)] = toks
    return q
