The per-stay h13 numbers explain the ×½ failure. Averaged over stays, insulin ×½ at h13 is about zero (−0.04), but the spread is wide and bimodal:
- about 25–29% of stays show a large glucose drop under halving (10% quantile −7.4, 25% quantile −6.0);
- the rest show the expected rise (+4 to +7).

The per-stay noise of the tool is small (shift_se ≈ 0.03 at h13), so this is a systematic, stay-specific artifact: for some stays the halved dose reads as more insulin. After h13 the ×½ shift keeps falling with each edited bundle (−0.04 → −5.5 by h24). Insulin ×2 also has a 10% tail on the wrong side (+1.5).

An effect confined to a subset of stays points to specific rate values. Halved or doubled rates in those stays may land in bins, or on edges, where my decoder misreads, possibly because the tool bins them differently from my local code.

Next I join the tool's per-stay h13 shifts with each stay's recorded hour-12 insulin and norepinephrine rates and doses from tuning.csv. I also compute the rate bins for r, r/2 and 2r under both edge conventions. If the wrong-sign stays share particular rate values or bin transitions, I have a concrete target for a fix. If they don't, the artifact comes from the state (c_hat side), not the encoding.

```python
tu = pd.read_csv('/data/tuning.csv')
p13 = dd[dd.hour == 13].pivot_table(index='stay_id', columns='edit', values='shift')
w12 = tu[(tu.hour >= 12) & (tu.hour < 13)]
ins = w12[w12.event == 'insulin'].groupby('stay_id').agg(r=('rate','first'), nI=('rate','size'), dI=('dose','first'), hI=('hour','first'))
ne = w12[w12.event == 'norepinephrine'].groupby('stay_id').agg(rn=('rate','first'), dn=('dose','first'))
T = p13.join(ins, how='left').join(ne, how='left')
def fb(x, side, attr='rate', code='insulin'):
    if not np.isfinite(x): return -1
    for c in (code, CID.get(code, None) if isinstance(CID, dict) else None):
        try: return int(fbin(c, attr, float(x), side=side))
        except Exception as e: err = repr(e)[:60]
    return err
T['bL'] = [fb(x, 'left') for x in T.r]; T['bR'] = [fb(x, 'right') for x in T.r]
T['hL'] = [fb(x/2, 'left') for x in T.r]; T['hR'] = [fb(x/2, 'right') for x in T.r]
T['dL'] = [fb(2*x, 'left') for x in T.r]; T['dR'] = [fb(2*x, 'right') for x in T.r]
Ti = T[T.r.notna()].sort_values('insulin_half')
cols = ['insulin_x2','insulin_half','r','nI','dI','hI','bL','bR','hL','hR','dL','dR']
print(Ti[cols].round(3).to_string(max_rows=80))
print('rate unique sample:', np.sort(tu[tu.event=='insulin'].rate.unique())[:25])
print('ne:', T[T.rn.notna()][['ne_x2','ne_half','rn','dn']].sort_values('ne_half').round(3).to_string(max_rows=16))
```