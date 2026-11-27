The offline pipeline is accurate. On the 38 stays shared with the tool's factual run, offline and tool shifts correlate 0.97–0.99 and the NLL changes are similar. For map_arterial they are +0.011 offline against +0.009 in the tool.

The blow-up comes from somewhere else. All the large map_arterial shifts are in stays with train ids (1010…), not tuning ids (2020…). So `RR` now mixes in rows from some other set of stays, whose `prev` indices point at the wrong states in `X`. That makes them garbage. Rows from the sel1 tuning stays are fine.

I'll check the `src` column and the id prefixes in `RR`. Then I'll redo the sel1 out-of-sample cost using only the tuning stays that match `X`, broken down by code and by whether a drug came in the previous bundle. That gives the honest per-reading cost of the current edit to compare with the tool's +0.0034.

```python
RRc['tun'] = RRc.stay >= 2_000_000_000
print(RRc.groupby(['tun','src']).size().to_string() if 'src' in RRc else RRc.tun.value_counts())
s1set = set(map(int, sel1))
RRc['ins1'] = RRc.stay.isin(s1set)
print('rows in sel1 stays', int(RRc.ins1.sum()), 'n sel1 stays present', RRc[RRc.ins1].stay.nunique())
R1m = RRc[RRc.ins1]
print('sel1-only offline dNLL saved edit %.4f (orig-vs-zero %.4f)' % (R1m.dn_s.mean(), (np.asarray(o_o[1])-nz_)[RRc.ins1.values].mean()))
print(R1m.groupby('code').agg(n=('dn_s','size'), dn=('dn_s','mean'), dpm_sd=('dpm_s','std'), maxabs=('dpm_s', lambda v: np.abs(v).max())).round(4).to_string())
```