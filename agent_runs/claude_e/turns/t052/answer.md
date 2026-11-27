The factual pass worked. The original model's value NLL on these 100 stays is 1.6346 per reading over 4,751 readings: finger glucose 1.464, lab 1.294, arterial MAP 1.471, cuff MAP 2.012. This is the baseline my edit must stay within 0.005 of.

I listed the stays as tuning ids 2020000000 to 2020000099, assuming that is sel1, the set whose states I saved. I need to check that assumption. If it is wrong, the factual output and the saved states cover different stays.

Next I will:
1. Check the ids against sel1.
2. Look at the factual file's columns.
3. Log the new findings in the notebook.

```python
ids_states=np.unique(S2['stay_id']); print("sel1 == 2020000000..99:", sorted(sel1)==list(range(2020000000,2020000100)), "| states stays match sel1:", set(ids_states)==set(sel1))
FA=pd.read_csv('/work/fwd_3/factual_original.csv'); print(FA.shape, list(FA.columns)); print(FA.head(3).to_string()[:500])
with open('/work/notes.md','a') as f: f.write("""
- Run 46-47: TRUE one-step coefs (gf_{t+1} on gf_t tier-FE quad, gf_{t-1}, lab_t; train). Raw dose lags1-6: [-2.01 -.74 1.40 1.28 1.22 1.15] (lags>=3 confounded).
  On dose DEVIATION (dose - E[dose|tier], tab E[dose|tier0..4,miss]=[0,1.03,2.07,3.24,6.24,.03]): [-2.01 -0.63 0.07 0.48 0.57 0.37]. Resid sd 12.8.
- fwd 3: factual original on tuning 2020000000..99: value NLL 1.63461 (n 4751); gf 1.4639, gl 1.2941, ma 1.4712, mc 2.012. File /work/fwd_3/factual_original.csv
""")
```