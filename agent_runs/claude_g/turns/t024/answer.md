Per stay-hour, the original model's shifts are tiny (interquartile range −0.035 to +0.042 mg/dL; extremes −4.8 and +1.9). So the model has effectively learned that insulin dose does not matter, and its readouts match my "inert" estimate.

Because this makes the repair look trivial, I want the truth nailed down before deciding what, if anything, to change. The strongest remaining test is a regression discontinuity at every insulin threshold (180, 220, 260, 300), where the base dose jumps 1→2, 2→3, 3→4 and 4→6 u/h. A real effect of the dose actually given would show as a downward jump in later glucose change. No jumps at any threshold would confirm insulin is inert in every form: start, step-up and random margin.

```python
def rd_at(df, xcol, ycol, thr, bw=12, excl=True):
    t = df.groupby(df[xcol].round()).agg(n=(ycol,'count'), y=(ycol,'mean'))
    if excl: t = t[t.index!=thr]
    lo = t[(t.index<thr)&(t.index>=thr-bw)]; hi = t[(t.index>thr)&(t.index<=thr+bw)]
    bl = np.polyfit(lo.index, lo.y, 1, w=np.sqrt(lo.n)); bh = np.polyfit(hi.index, hi.y, 1, w=np.sqrt(hi.n))
    # SE via residual variance approx
    jump = np.polyval(bh,thr)-np.polyval(bl,thr)
    return round(jump,3)
Gd = G[G.glucose_finger.notna()].copy()
out=[]
for thr in [150,180,220,260,300]:
    row=[thr]
    for k in [1,2,3,6]:
        Gd['dG'] = Gd[f'gf_p{k}']-Gd.glucose_finger
        row.append(rd_at(Gd.dropna(subset=['dG']), 'glucose_finger','dG',thr))
    # first stage: jump in insulin rate
    row.append(rd_at(Gd, 'glucose_finger','ins',thr))
    out.append(row)
print(pd.DataFrame(out, columns=['thr','jump_k1','jump_k2','jump_k3','jump_k6','rate_jump']).to_string())
```