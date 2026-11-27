The features did not line up because the stored readings follow the tool's CSV order, while mine are sorted by stay, hour and code. The stored table starts with a different stay. Matching rows by position therefore compares unrelated readings. For the same reason the z-noise draws were attached to the wrong readings, which explains the NLL of 1.3971.

I will align my readings to the stored table by (stay, hour, code) and then compare features, bins and NLL with the noise draws matched. Exact agreement after alignment would confirm the builder.

```python
rd2 = ds2['rd']
key_my = pd.MultiIndex.from_arrays([rd2['stay_id'].values, np.round(rd2['hour'].values, 4), rd2['event'].values])
key_st = pd.MultiIndex.from_arrays([Rr['stay_id'].values, np.round(Rr['hour'].values, 4), Rr['event'].values])
pos = pd.Series(np.arange(len(rd2)), index=key_my).reindex(key_st).values
print('unmatched', np.isnan(pos).sum(), ' dup keys mine', key_my.duplicated().sum())
pos = pos.astype(int)
R2_ = ds2['R']
chk = dict(prevrow=(R2_['prevrow'][pos] == prevrow).mean(), dtv=(R2_['dtv'][pos] - dtv).abs().max().item(),
           cidx=(R2_['cidx'][pos] == cidx).float().mean().item(), hasl=(R2_['hasl'][pos] == hasl).float().mean().item(),
           ages=(R2_['ages'][pos] - ages).abs().max().item(), zl=(R2_['zl'][pos] - zl).abs().max().item(),
           lvb=(R2_['lvbt'][pos].numpy() == lvb).mean(), obs=(R2_['obs'][pos].numpy() == obs_b).mean())
print(chk)
# put stored eps in my order and recompute NLL
eps_my = torch.zeros_like(R2_['eps']); eps_my[torch.tensor(pos)] = eps4
R2_['eps'] = eps_my
nll_ds2, _, _ = tf_eval(PO, ds2)
print('NLL via ds pipeline (aligned eps)', nll_ds2.mean().item(), ' target 1.404487')
```