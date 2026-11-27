## Step 22: most "starts" above 65 are patients arriving on norepinephrine

**What the last result showed.**
- Most apparent starts with MAP above 65 happen at hour 0. Those patients arrive already on norepinephrine; it is not recorded before ICU admission, so the previous rate looks like 0 when it is not.
- The start rule therefore looks like: latest MAP (arterial preferred) below 65 means start.
- Starting rates are low (median 0.10, mode 0.04–0.05) but widely spread.

**What I will do next.**
- Re-check starts excluding hour 0.
- Study titration among hours where norepinephrine was on in the previous hour: the rate change (absolute and in logs) against the latest MAP, binned.

**What would count as evidence.**
- A rule such as "multiply the rate up when MAP < 65 and down when MAP > 75", with random step sizes, would give titration surprises for an instrument.
- A deterministic step would push me to use the start threshold at 65 (a discontinuity design) or the cap at 1.0 instead.

```python
st2 = xn[(xn.ne_prev==0) & (xn.h>0) & xn.lm_art_pref.notna()]
print("START (h>0) by art-pref latest MAP:", st2.groupby(pd.cut(st2.lm_art_pref,[0,60,63,64,65,66,67,70,75,200])).ne_rate.apply(lambda s:(s>0).mean()).round(4).to_dict())
on = xn[(xn.ne_prev>0) & xn.lm_art_pref.notna()].copy()
on['lr'] = np.log(np.where(on.ne_rate>0, on.ne_rate, np.nan)/on.ne_prev)
on['mb'] = pd.cut(on.lm_art_pref,[0,50,55,60,62,64,65,66,68,70,72,75,78,80,85,90,200])
print(on.groupby('mb').agg(n=('ne_rate','size'), stop=('ne_rate',lambda s:(s==0).mean()), dmean=('dne','mean'), dsd=('dne','std'), lr_mean=('lr','mean'), lr_sd=('lr','std')).round(3).to_string())
```