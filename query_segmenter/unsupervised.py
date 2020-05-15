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
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


class Segmenter(object):
    """Implementation of Unsupervised Query Segmentation."""

    def __init__(self):
        """Construct Segmenter."""
        self.scores = defaultdict(float)
        self.stats = defaultdict(lambda: self._initialize_ngram())

    def compute_scores(self, queries, alpha=1, beta=0.0):
        """Compute segmentation scores.

        Parameters
        ----------
        queries : list (n_queries)
            Training data for determining segments.
        alpha : int, optional
            Minimum query frequency. All words in an n-gram must occur in
            at least alpha queries to be considered a "significant n-gram."
        beta : float, optional
            beta * k is the segmentation score threshold, where k is the number
            of queries that contains all the words of a given n-gram.

        Returns
        -------
        self.scores : dict
            A mapping of ngram to segmentation score.

        Algorithm
        ---------
        Consider an n-gram M = [word_1, word_2, ..., word_n].

        Let {query_1, query_2, ..., query_k} be the subset of queries
        that contain all the words of M in some order.

        To test if M is a multi-word entity, we check if M occurs
        together more often than not.

        """
        sig_ngrams = self._get_significant_ngrams(queries, alpha=alpha)
        self._precompute_stats(sig_ngrams, queries)

        for ngram in sig_ngrams:
            N = self.stats[ngram]['frequency']
            k = self.stats[ngram]['co_occur']
            E_X = self.stats[ngram]['expectation']

            score = 2 * (N - E_X)**2 / k

            if score < beta * k:
                # if the score does not exceed threshold
                continue

            self.scores[ngram] = score

        return self.scores

    def segment(self, query):
        """Segment a query (single or batch).

        Parameters
        ----------
        query : str or array-like

        Returns
        -------
        segments : 1 or 2D list of strings

        """
        if isinstance(query, str):
            segments = self._segment(query)
        elif hasattr(query, '__iter__'):
            segments = list(map(self._segment, query))
        return segments

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
        """Precompute counts for each ngram M.

        The segmentation score is a function of three values:
            - the number of times M occurs in any order ( k )
            - the number of times M occurs in that order ( N )
            - the total probability that M occurs in any order ( E(X) )

        A high score indicates that M is a Multi-Word Entity.
        A low score indicates that M is not a Multi-Word Entity.

        """
        for ngram in sig_ngrams:
            ngram_len = len(ngram.split())
            for query in queries:
                query = re.sub('[^a-zA-Z0-9]', ' ', query)  # unless special characters are replaced with space the tokenization wont match
                sub_queries = query.split()
                if all(word in sub_queries for word in ngram.split()):
                    # if all words in n-gram co-occurs in query
                    self.stats[ngram]['co_occur'] += 1
                    if self._match_ngram(ngram, query):
                        # if the n-gram occurs in query
                        self.stats[ngram]['frequency'] += 1

                    query_len = len(query.split())
                    numer = math.factorial(max(query_len - ngram_len + 1,0))   # query_len - ngram_len + 1 can possibey be negative, thus clamping to zero
                    denom = math.factorial(query_len)
                    prob = numer / denom

                    self.stats[ngram]['expectation'] += prob

        return self.stats

    def _initialize_ngram(self):
        """Initialize the information to collect about each ngram."""
        return {
            'frequency': 0,
            'co_occur': 0,
            'expectation': 0
        }

    def _match_ngram(self, ngram, query):
        match = re.search(rf'\b{ngram}\b', query)
        return match is not None

    def _segment(self, query):
        """Segment a query with dynamic programming."""
        query = query.split()
        query_len = len(query)
        val = [0] * (query_len+1)
        idx = [0] * (query_len+1)
        for i in range(1, query_len+1):
            max_val = -1
            for j in range(i):
                subquery = ' '.join(query[j:i])

                if subquery in self.scores:
                    score = self.scores[subquery]+val[j]
                else:
                    score = val[j]

                if score >= max_val:
                    max_val = score
                    idx[i] = j
            val[i] = max_val

        segments = []
        while query_len > 0:
            query_len = idx[query_len]
            segments.insert(0, ' '.join(query[query_len:]))
            query = query[:query_len]

        return segments
