The CSV does contain every measurement, so the register history is complete. The error has a clear pattern: it correlates −0.56 with the CSV's predicted mean, with a −5.3 coefficient on the last-value z-score. My reproduction is therefore less persistent than the real model, pulled too far toward the centre. The last-value pathway (z column, bin embedding or transition term) is probably under-applied or mis-specified. Below I test scaling and dropping each of those components to see which one closes the gap.

```python
def predict_reg(FR, zmul=1.0, drop_bins=False, drop_trans=False, trans_mul=1.0, binmul=1.0):
    W = dict(Wd)
    if drop_bins: W['embed.bins.value.weight'] = np.zeros_like(Wd['embed.bins.value.weight'])
    else: W['embed.bins.value.weight'] = Wd['embed.bins.value.weight']*binmul
    W['embed_transition.value_bin.weight'] = Wd['embed_transition.value_bin.weight']*(0 if drop_trans else trans_mul)
    lnw = Wd['last_numeric.weight'].copy(); lnw[:,2]*=zmul; W['last_numeric.weight']=lnw
    return predict(FR,typ=1,gap_scale=10.0,W=W)
for tag,kw in [('z1.5',dict(zmul=1.5)),('z2',dict(zmul=2.0)),('trans1.5',dict(trans_mul=1.5)),('trans2',dict(trans_mul=2.0)),('bins2',dict(binmul=2.0)),('nobins',dict(drop_bins=True)),('notrans',dict(drop_trans=True))]:
    score(predict_reg(FR,**kw),tag)
print(np.linalg.norm(Wd['last_numeric.weight'],axis=0).round(2))
```