On tu2 (2020000300..399) the −2.6 edit costs only +0.0010 per reading: glucose_finger +0.006 per row, glucose_lab −0.004, MAP unchanged. tu0 at the same gain cost +0.0043. So the cost swings a lot between batches, depending on how much insulin each batch contains. The averages are about 0.0032 at −2.6 and 0.0028 at −2.5. What matters is the chance that an unseen evaluation set exceeds 0.005.

I will estimate that by resampling stays. From the per-reading NLLs of all four tuning batches I compute each stay's cost:
- tu0 and tu2 are measured at −2.6;
- f3 and tu1 are measured at −2.5, and I scale them by about 1.12 to approximate −2.6.

I then draw evaluation sets of 100–1000 stays and compute the per-reading ratio. A risk below about 1% for 400+ stays would justify keeping −2.6. A clearly higher risk means going back to −2.5.

```python
def stay_cost(k):
    o=pd.read_csv(f'/work/fwd_{k}/factual_original.csv'); e=pd.read_csv(f'/work/fwd_{k}/factual_edited.csv')
    return o,e
o17,e17=stay_cost(17); print(list(o17.columns)); print(o17.head(2).to_string())
```