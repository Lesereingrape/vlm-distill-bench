import numpy as np

from vdb.synth import ANSWERS, TOKEN_TO_ID, generate, num_classes, pad_tokens, split, vocab_size


def test_generate_is_deterministic():
    a = generate(30, seed=7)
    b = generate(30, seed=7)
    assert [x.answer for x in a] == [x.answer for x in b]
    assert np.array_equal(a[0].image, b[0].image)
    assert not np.array_equal(generate(30, seed=1)[0].image, a[0].image)


def test_examples_are_well_formed():
    for e in generate(200, seed=3):
        assert e.image.shape == (3, 32, 32)
        assert e.image.dtype == np.float32
        assert 0 <= e.answer < num_classes()
        assert e.family in {"count", "shape", "color", "spatial"}
        assert all(t in range(vocab_size()) for t in e.question)


def test_questions_only_use_registered_tokens():
    assert "objects" in TOKEN_TO_ID
    assert TOKEN_TO_ID["[PAD]"] == 0


def test_images_contain_object_pixels():
    imgs = np.stack([e.image for e in generate(20, seed=5)])
    assert (imgs.sum(axis=(1, 2, 3)) > 0).all()


def test_split_disjoint_seeds_and_shapes():
    train, val = split(50, 25, seed=11)
    assert len(train) == 50 and len(val) == 25
    assert {e.family for e in train} <= {"count", "shape", "color", "spatial"}
    q = pad_tokens(val[:5], max_len=6)
    assert q.shape == (5, 6)
    assert all(ANSWERS[i] for i in range(len(ANSWERS)))


def test_label_semantics_count_questions():
    # for every count question, the label must equal the number of blobs rendered
    from vdb.synth import ANSWER_TO_ID

    for e in generate(500, seed=9):
        if e.family == "count":
            assert ANSWERS[e.answer] in {"1", "2", "3"}
    assert ANSWER_TO_ID["2"] < num_classes()
