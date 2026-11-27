Under the tool's four-draw scoring (offline base NLL 1.5955), several units are still net harmful. Ablating unit 31 lowers NLL by 0.0011, unit 143 by 0.0011, unit 54 by 0.0007 and unit 9 by 0.0006. The model was presumably trained with posterior z, so under prior z draws these units add slightly more noise than signal.

That makes repurposing nearly free, and may even buy some NLL headroom. Before committing, I am checking:

1. the combined effect of ablating units 31, 143, 54 and 9 together (savings may not add up);
2. how cleanly a linear gate separates MAP from glucose rows in with_code, under z noise (the channel needs the gate far from zero);
3. the bin-centre tables, so I can fit per-family logit tilts;
4. whether my two insulin-surprise definitions (panel-based vs reading-table-based) agree, since they gave lag-1 deficits of -1.3 vs -1.5.

```python
abl = [31,143,54,9]
lga = lgd.copy()
for j in abl: lga = lga - (Hd[:,:,j]-mh4[j])[:,:,None]*W2[:,j][None,None,:]
print('joint ablation dNLL', round(float(nll_avg(lga).mean()-base4),5))
y_g = np.where(ism, 1.0, -1.0)
Xg = np.hstack([WC0, np.ones((len(WC0),1))])
vg = np.linalg.lstsq(Xg, y_g, rcond=None)[0]
gd = WCd@vg[:192] + vg[192]
print('gate on MAP rows min/p1', round(float(gd[:,ism].min()),3), round(float(np.percentile(gd[:,ism],1)),3), ' on gluc rows max/p99', round(float(gd[:,isg].max()),3), round(float(np.percentile(gd[:,isg],99)),3))
print('vocab keys', list(VOCAB.keys()))
print('corr eps defs', round(float(np.corrcoef(RD2['ei1'].values[rg], RD2['eps_ins_1'].fillna(0).values[rg])[0,1]),3))
```