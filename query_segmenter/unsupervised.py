"""Implementation of Unsupervised Query Segmentation.

- author: @kchro

Unsupervised Query Segmentation Using only Query Logs [Mishra et. al. 2011]

https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/pp0295-mishra.pdf

Usage:
    from query_segmenter.unsupervised import Segmenter

    segmenter = Segmenter()
    segmenter.compute_scores(queries)
    segments = segmenter.segment('new iphone 6')
    assert segments == ['new', 'iphone 6']

"""


class Segmenter(object):
    """Implementation of Unsupervised Query Segmentation."""

    def __init__(self):
        """Construct Segmenter."""
        pass

    def compute_scores(self, queries):
        """Compute segmentation scores."""
        pass

    def segment(self, query):
        """Segment a new query."""
        pass

