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
import math
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer


class Segmenter(object):
    """Implementation of Unsupervised Query Segmentation."""

    def __init__(self):
        """Construct Segmenter."""
        self.scores = defaultdict(float)
        self.stats = {}

    def compute_scores(self, queries, alpha=0):
        """Compute segmentation scores.

        Algorithm:

        Consider an n-gram M = [word_1, word_2, ..., word_n].

        Let {query_1, query_2, ..., query_k} be the subset of queries
        that contain all the words of M in some order.

        To test if M is a multi-word entity, we check if M occurs
        together more often than not.

        """
        sig_ngrams = self._get_significant_ngrams(queries, alpha=alpha)
        self._precompute_stats(sig_ngrams, queries)

        for ngram in sig_ngrams:
            frequency = self.stats[ngram]['frequency']
            co_occur = self.stats[ngram]['co_occur']
            expectation = self.stats[ngram]['expectation']

            score = (frequency - expectation)**2 / co_occur
            self.scores[ngram] = score

        return self.scores

    def segment(self, query):
        """Segment a new query."""
        pass

    def _get_significant_ngrams(self, queries, ngram_range=(2, 10), alpha=1):
        """Get significant n-grams in the list of queries.

        An n-gram must occur at least "alpha" times to be significant.
        CountVectorizer

        """
        cv = CountVectorizer(
            ngram_range=ngram_range,
            token_pattern=r"\w+",
            min_df=alpha)

        cv.fit(queries)
        sig_ngrams = list(cv.vocabulary_.keys())
        return sig_ngrams

    def _precompute_stats(self, sig_ngrams, queries):
        """Precompute counts for each ngram.

        The segmentation score is a function of three values:
            - the number of times M occurs in any order ( k )
            - the number of times M occurs in that order ( N )
            - the total probability that M occurs in any order ( E(X) )

        A high score (-log delta) indicates that M is a Multi-Word Entity.
        A low score (-log delta) indicates that M is not a Multi-Word Entity.

        """
        for ngram in sig_ngrams:
            if ngram not in self.stats:
                self._initialize_ngram(ngram)

            ngram_len = len(ngram.split())
            for query in queries:
                sub_queries = query.split()
                if all(word in sub_queries for word in ngram.split()):
                    # if all word in ngram co-occurs in query
                    self.stats[ngram]['co_occur'] += 1
                    if self._match_ngram(ngram, query):
                        self.stats[ngram]['frequency'] += 1

                    query_len = len(query.split())
                    prob = math.factorial(query_len - ngram_len + 1) / \
                           math.factorial(query_len)

                    self.stats[ngram]['expectation'] += prob

        return self.stats

    def _initialize_ngram(self, ngram):
        """Initialize the information to collect about each ngram."""
        self.stats[ngram] = {
            'frequency': 0,
            'co_occur': 0,
            'expectation': 0
        }

    def _match_ngram(self, ngram, query):
        match = re.search(rf'\b{ngram}\b', query)
        return match is not None
