setA is 100 tuning stays (ids 20200000xx). Two things stand out:
- fwd_1 is the original model's dose_edit on setA.
- fwd_7 holds a factual "both" result that the notebook doesn't record.

Since 7 passes are used, fwd_7 was probably the factual check of the saved v4 edit, and its output was the message that got blocked. I'll read it from disk instead of spending another pass. I'll also take the original model's dose-edit readouts from fwd_1 to compare against the record-based targets (−14.17 / +7.09 / +4.82 / −2.41).

If the edited NLL is within about 0.002 of the original, factual fit is fine and the next step is a dose_edit "both" pass to calibrate gains. If not, the edit must be shrunk.

```python
for k in [1,7]:
    s=json.load(open(f'/work/fwd_{k}/summary.json')); print(k, json.dumps(s)[:900])
fo=pd.read_csv('/work/fwd_7/factual_original.csv'); fe=pd.read_csv('/work/fwd_7/factual_edited.csv')
print(fo.shape, fe.shape, list(fo.columns))
nl=[c for c in fo.columns if 'nll' in c.lower()]; mu=[c for c in fo.columns if 'mean' in c.lower()]
ev=[c for c in fo.columns if ('event' in c.lower() or 'code' in c.lower())]
print(nl, mu, ev)
print('stays==setA', set(fo['stay_id'].unique())==set(int(i) for i in ids), fo['stay_id'].nunique())
d=fo[[ev[0]]].copy(); d['no']=fo[nl[0]]; d['ne']=fe[nl[0]]; d['dmu']=fe[mu[0]]-fo[mu[0]]
g=d.groupby(ev[0]).agg(n=('no','size'),no=('no','mean'),ne=('ne','mean'),dmu=('dmu','mean'),sdmu=('dmu','std'))
g['dnll']=g.ne-g.no; print(g.round(4).to_string())
print('overall dnll', round(d.ne.mean()-d.no.mean(),5), 'max|dmu|', round(d.dmu.abs().max(),2))
```