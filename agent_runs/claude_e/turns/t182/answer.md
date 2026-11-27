RR combines two sources: 5,214 rows from tuning stays (src '1', states in fwd_2) and 5,139 rows from training stays (src 'A', states in fwd_4). Both are held out from the readout and gate fits. Each row's 'prev' indexes its own source's states.

I'll redo the gate check with each row's previous-bundle flags taken from the matching states file, so the misclassification numbers mean what they claim.

```python
z4 = np.load('/work/fwd_4/states_original.npz', allow_pickle=True)
src = RR.src.values
def from_src(key):
    out = np.zeros(len(RR), dtype=z2[key].dtype)
    m1 = src=='1'; mA = src=='A'
    out[m1] = z2[key][prev[m1]]; out[mA] = z4[key][prev[mA]]
    return out
hi_s = from_src('has_insulin').astype(bool); pv_ne = from_src('has_norepinephrine').astype(bool); pv_na = from_src('n_atoms')
GridFor = lambda s: Gtu if str(s).startswith('202') else Gtr
def ins_at(s, h):
    G_ = GridFor(s)['ins']
    return (G_.at[s,h] if (s in G_.index and h in G_.columns and pd.notna(G_.at[s,h])) else 0) > 0
ins_h1 = np.array([ins_at(s, int(h)-1) for s,h in zip(RR.stay.values, RR.hour.values)])
ins_h2 = np.array([ins_at(s, int(h)-2) for s,h in zip(RR.stay.values, RR.hour.values)])
on_s = gl_rows & hi_s; off_s = gl_rows & ~hi_s
print('glucose rows: after insulin bundle %d, other %d' % (on_s.sum(), off_s.sum()))
print('gate on-rows pct1/5/50 %s | off-rows pct50/95/99 %s' % (np.percentile(G_[:,on_s],[1,5,50]).round(1), np.percentile(G_[:,off_s],[50,95,99]).round(1)))
print('on-rows mean|P-R| %.3f ; off-rows mean|P| %.3f' % (np.abs(P_-R_)[:,on_s].mean(), np.abs(P_)[:,off_s].mean()))
Gm = G_.mean(0)
dfm = pd.DataFrame({'on': hi_s, 'gate_on': Gm > 0, 'ne_prev': pv_ne, 'natoms': pv_na, 'dt': RR.dt.values, 'ins_h1': ins_h1, 'ins_h2': ins_h2, 'lab': cs==10})[gl_rows]
print(dfm.groupby(['on','gate_on']).agg(n=('dt','size'), ne_prev=('ne_prev','mean'), natoms=('natoms','mean'), dt=('dt','median'), ins_h1=('ins_h1','mean'), ins_h2=('ins_h2','mean'), lab=('lab','mean')).round(3).to_string())
```