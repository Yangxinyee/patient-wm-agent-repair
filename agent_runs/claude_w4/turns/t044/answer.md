The vocab adds one important detail: the predicted mean is a binned mean, the sum of p(b) times the bin centre over 16 coarse bins. For glucose_finger the top bin is everything above 258, centred at 269. For very high glucose the model's mean saturates, so a large true drop may show up compressed.

A key diagnostic comes before choosing a design: does the old model already account for within-band dose surprises in factual mode?
- If the factual residuals (reading minus predicted mean) are negatively related to recent insulin surprises at about −2 per unit/h, the model ignores doses. Adding surprise effects would then also improve the factual NLL, which gives slack under the 0.005 limit.
- If the residuals are unrelated to surprises, the model handles within-band surprises but fails to extrapolate to ×2 doses. That calls for a different fix.

First I need to recover my kernel's helper functions and variables (their source is not retrievable) and look at the factual CSV layout. I will print signatures and docstrings, the types and shapes of the key variables, and the head of the factual output.

```python
import inspect
def desc(v):
    if isinstance(v, (pd.DataFrame,)): return f"DataFrame{v.shape} cols={list(v.columns)[:25]}"
    if isinstance(v, pd.Series): return f"Series{v.shape} name={v.name}"
    if isinstance(v, np.ndarray): return f"ndarray{v.shape} {v.dtype} " + (str(np.round(v,3)) if v.size<=30 else '')
    if isinstance(v, (list, tuple)): return f"{type(v).__name__}[{len(v)}] " + str(v[:6])[:200]
    if isinstance(v, dict): return f"dict[{len(v)}] keys={list(v.keys())[:12]}"
    if callable(v):
        try: return 'fn' + str(inspect.signature(v)) + ' doc=' + str(v.__doc__)[:150]
        except Exception as ex: return 'callable'
    return f"{type(v).__name__}: {str(v)[:120]}"
for k in ['tr','tu','P','PT','PTr','g','gT','panel','wide','xi','dn','di','featsI','featsN','irf','irf_ne','hF','hA','hL12','bL','bF','bA','seA','sel','sel100','both_ids','info','chk2','mult','ks','re','S','tsid','gt','fw','true_shift','tsls','crossfit','first_dose_from12','add_last','band_of','rate_band','latest','pref','base_rate','band_mean','rates','art_stay','y','z','base','res']:
    if k in globals(): print(k, '->', desc(globals()[k]))
fo = pd.read_csv('/work/fwd_2/factual_original.csv'); print(fo.shape, list(fo.columns)); print(fo.head(4).to_string())
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de.shape, list(de.columns)); print(de.head(4).to_string())
```