"""Unit tests for the query segmenter."""
import sys
import os
import warnings
from pathlib import Path
try:
    from query_segmenter.unsupervised import Segmenter
except ModuleNotFoundError:
    project_dir = str(Path(__file__).parent.parent)
    sys.path.append(project_dir)
    from query_segmenter.unsupervised import Segmenter


def suppress_warnings(func):
    def wrapper():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func()
    return wrapper


@suppress_warnings
def test_get_significant_ngrams():
    segmenter = Segmenter()
    output = segmenter._get_significant_ngrams([
        'iphone 6',
        'iphone 6',
        'iphone 6 red',
        'iphone 7',
        'new iphone 7',
        'iphone 7 black'
    ], alpha=2)

    assert output == ['iphone 6', 'iphone 7']

@suppress_warnings
def test_precompute_stats():
    segmenter = Segmenter()
    queries = [
        'iphone 7',
        'new iphone 7'
    ]

    sig_ngrams = segmenter._get_significant_ngrams(queries, alpha=2)
    assert sig_ngrams == ['iphone 7']

    output = segmenter._precompute_stats(
        sig_ngrams=sig_ngrams,
        queries=queries)

    expected = {
        'iphone 7': {
            'frequency': 2,
            'co_occur': 2,
            'expectation': 0.8333333333333333
        }
    }

    assert output == expected

@suppress_warnings
def test_compute_scores():
    segmenter = Segmenter()
    output = segmenter.compute_scores([
        'iphone 6',
        'iphone 6',
        'iphone 6 red',
        'iphone 7',
        'new iphone 7',
        'iphone 7 black'
    ], alpha=2)

    expected = 2.240740740740741
    assert output['iphone 7'] == expected


@suppress_warnings
def test_segment():
    segmenter = Segmenter()
    output = segmenter.compute_scores([
        'iphone 6',
        'iphone 6',
        'iphone 6 red',
        'iphone 7',
        'new iphone 7',
        'iphone 7 black'
    ], alpha=2)

    # single string
    output = segmenter.segment('iphone 6 charger')
    expected = ['iphone 6', 'charger']
    assert output == expected


@suppress_warnings
def test_segment_batch():
    segmenter = Segmenter()
    output = segmenter.compute_scores([
        'iphone 6',
        'iphone 6',
        'iphone 6 red',
        'iphone 7',
        'new iphone 7',
        'iphone 7 black'
    ], alpha=2)

    output = segmenter.segment([
        'iphone 6 in box',
        'iphone 6 charger',
        'new iphone x',
        '$700 iphone 7'
    ])

    expected = [
        ['iphone 6', 'in', 'box'],
        ['iphone 6', 'charger'],
        ['new', 'iphone', 'x'],
        ['$700', 'iphone 7']
    ]

    assert output == expected


@suppress_warnings
def test_segment_toy(dirname='tests/test_resources/toy'):
    with open(os.path.join(dirname, 'queries.txt'), 'r') as f:
        queries = [line.strip() for line in f]
    segmenter = Segmenter()
    segmenter.compute_scores(queries)

    with open(os.path.join(dirname, 'test_queries.txt'), 'r') as f:
        expected = [line.strip().split(',') for line in f]

    output = segmenter.segment(queries)
    for pred, gold in zip(output, expected):
        print(pred, gold)

    assert output == expected