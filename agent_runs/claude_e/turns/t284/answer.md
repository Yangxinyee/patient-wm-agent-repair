The cost on tu1 at -2.7 is +.00429, confirming the edit hash (2bc751d207). That matches the prediction from its -2.5 cost scaled by gain² (.00369 × 1.166 = .0043). So cost scaling as gain² is now confirmed on two batches. The two most expensive tuning batches sit at .0046 and .0043; the two cheaper ones are about .0034 and .0011.

The remaining question is risk. My quick batch-level estimate puts P(cost > .005) at -2.7 at about 2% for a 400-stay evaluation set. The earlier stay-level bootstrap gave about 0.5%.

To settle this I will redo a stay-level bootstrap with every factual pass of the insulin-only edit. That is 4 tuning and 4 training batches, 800 stays, with each stay's cost scaled to the target gain by gain². I will report P(>.005) at gains -2.5, -2.6, -2.65 and -2.7 for evaluation sets of 100–800 stays, both for all stays and for tuning stays only, since unseen stays may resemble tuning more than training. This decides whether to keep -2.7 or step back slightly.

```python
import glob
info = []
for k in range(1, 30):
    fo_, fe_ = f'/work/fwd_{k}/factual_original.csv', f'/work/fwd_{k}/factual_edited.csv'
    if not (os.path.exists(fo_) and os.path.exists(fe_)): continue
    a = pd.read_csv(fo_); b = pd.read_csv(fe_)
    d = b['nll'].values - a['nll'].values
    info.append((k, a.stay_id.min(), a.stay_id.max(), a.stay_id.nunique(), round(d.mean(),5)))
print(pd.DataFrame(info, columns=['fwd','smin','smax','n','cost']).to_string())
```