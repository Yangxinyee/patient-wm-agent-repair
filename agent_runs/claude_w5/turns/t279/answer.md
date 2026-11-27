fwd_5 shows that the tool does run my edit in factual mode: states differ by 0.7% (insulin rows) and 1.4% (norepinephrine rows) relative. The z-posterior changes by 11% at reading bundles. That is expected if the belief about the current reading shifts, since z encodes the surprise of each new reading.

I had already stored a per-bundle comparison of local and tool states in `recs`, with fields err_o, err_e, nd_tool and nd_loc. Summarising it now will show:
- whether the edited-model mismatch (err_e) is far larger than the original's (err_o);
- whether the local and tool sizes of the edit's effect (nd_loc vs nd_tool) agree.

If nd_loc and nd_tool agree, my local edited model is faithful. The failure is then a design problem: bias in the decoded residual and no dose response, which I can then study entirely locally. If they disagree, the tool's computation differs from mine.

```python
import numpy as np, pandas as pd
R_ = pd.DataFrame(recs)
R_['drugc'] = R_['drug'].astype(str)
print(R_.head(3).to_string())
print(R_.groupby('drugc')[['err_o','err_e','nd_tool','nd_loc']].median())
for dc, g in R_.groupby('drugc'):
    a, b = np.asarray(g.nd_tool, float), np.asarray(g.nd_loc, float)
    print(dc, 'corr(nd_tool, nd_loc)=%.3f  median ratio loc/tool=%.3f' % (np.corrcoef(a, b)[0,1], np.median(b/np.maximum(a,1e-12))))
R2_ = pd.DataFrame(recs2); R2_['drugc']=R2_['drug'].astype(str)
print(R2_.groupby('drugc')[['err','err_out','err_z','nrm']].median())
print('SOL keys', {k: (type(v).__name__, getattr(v,'shape',None)) for k,v in SOL.items()})
```