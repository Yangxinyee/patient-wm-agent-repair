Run 50 explains the gap. At capped rates EPSN still "predicts" MAP (RF 4.25) while the rate does not move at all. So for capped rows EPSN is not noise: it encodes the MAP band, and capping also selects high noise draws. That contaminated my earlier NE IV estimates, including gam_ne in effects.json.

On uncapped rows, the per-log-unit effect is about 5 mmHg and the per-rate effect about 8.5–8.7. Even those are rough, because dropping capped rows truncates the noise.

The fix is a cleaner instrument. I'll keep only titration hours whose pre-decision expected rate, rate_{t−1}·f(MAP_t), sits in an interior band where capping or stopping is very unlikely. Within that band I'll residualise the log-ratio on the pre-decision variables, so the instrument is mean-zero given history by construction. First I need the MAP the NE policy actually decides on (Mdec) and to confirm the f steps. I'll also record the finding in the notebook.

```python
with open('/work/notes.md','a') as f: f.write("""
- Run 44: insulin lag-1 effect per u/h constant across base levels (-1.9..-2.3), Z^2 small -> insulin effect LINEAR additive. ins x2:x1/2 ~ -2:1.
- Run 46/50: NE effect per unit rate falls with rate (10.8,7.3,5.7,4.6 by quartile) but per log-rate ~constant (2.7-3.5 lag1) -> NE ~ LOG-linear (or Emax).
  WARNING: EPSN is not clean at capped rates (encodes MAP band; capping selects high noise) -> gam_ne in effects.json is BIASED. Need interior-stratum instrument.
""")
ok_ = ~np.isnan(Mdec)
print('Mdec==MA where MA present: %.3f ; Mdec==MC where MA absent: %.3f' % (
    np.mean((Mdec==MA)[~np.isnan(MA)&ok_]), np.mean((Mdec==MC)[np.isnan(MA)&~np.isnan(MC)&ok_])))
print('frac art stays', art_stay.mean(), ' MA present frac in art stays', np.mean(~np.isnan(MA[art_stay.astype(bool)])))
# titration factor by Mdec band for interior rows
ok = (NOR[:,1:25]>0)&(NOR[:,0:24]>0.15)&(NOR[:,0:24]<0.45)&~np.isnan(Mdec[:,1:25])
lr_ = np.log(NOR[:,1:25][ok]/NOR[:,0:24][ok]); md_ = Mdec[:,1:25][ok]
for lo_,hi_ in [(0,62),(62,65),(65,68),(68,72),(72,75),(75,78),(78,200)]:
    s_=(md_>=lo_)&(md_<hi_); print('Mdec %d-%d n=%d mean %.3f sd %.3f'%(lo_,hi_,s_.sum(),lr_[s_].mean(),lr_[s_].std()))
```