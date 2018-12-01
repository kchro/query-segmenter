# query-segmenter
Query Segmentation for search

`query_segmenter.unsupervised.Segmenter`:
  - an implementation of [Unsupervised Query Segmentation Using only Query Logs](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/pp0295-mishra.pdf) [Mishra et. al. 2011] from Microsoft Research.

Install:


Usage:
```python
from query_segmenter.unsupervised import Segmenter

queries = [
  "iphone 6",
  "new iphone 6",
  "samsung galaxy s8",
  "old iphone 6",
  "iphone 7",
  "samsung galaxy note",
  ...
]

segmenter = Segmenter()
segmenter.compute_scores(queries)
segments = segmenter.segment('iphone 6 case')
print(segments)

>> ['iphone 6', 'case']
```
