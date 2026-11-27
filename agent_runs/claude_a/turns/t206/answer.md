The capture works. `tf_eval` calls `head_value` once for all 14,296 readings in dsE, and the value-MLP input `with_code` comes through as a 14296 × 4 × 192 tensor (4 prior z samples). The error was only my indexing: `dsE['rd']` is a DataFrame.

Next I will capture `with_code` under the original model and find the code column of `rd`. Then I will fit a least-squares discriminant of glucose_finger versus the other value codes. I will report the margin: the lowest glucose_finger score against the highest other-code score, over every z sample. The fit is on half the readings and the check on the other half, so I do not fool myself.

How I will read the result:
- A clean, wide margin (no overlap) means a code gate needs no extra killed dim or encoder cut.
- Overlapping scores mean I fall back to the encoder-cut gate dim.

```python
rd=dsE['rd']; print('rd cols', list(rd.columns))
CW={}
def _cap2(x, pre):
    if pre=='attr_heads.value_bin': CW['wc']=x.detach().clone()
    if pre=='z_prior': CW['ctx']=x.detach().clone()
    return _orig_mlp2(x, pre)
globals()['mlp2']=_cap2
try: r_=tf_eval(PO, dsE, False)
finally: globals()['mlp2']=_orig_mlp2
WC=CW['wc']; print('wc', tuple(WC.shape), 'nll', float(r_[0].mean()))
cc=[c for c in rd.columns if rd[c].dtype.kind in 'iu' and rd[c].nunique()<=13]
print('int cols', {c:sorted(rd[c].unique().tolist())[:8] for c in cc})
```