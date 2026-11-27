The arrays I needed are there. `mA2` uses my fitted layout for the 8 inputs of `last_numeric`. It matches the tool's predicted means with MAE 1.26 and only small biases (+0.4 finger glucose, −0.2 lab, −0.1 arterial, +0.3 cuff). The remaining error is probably mostly noise from the tool's 4 random prior draws of z. The older WC99 used a worse layout, which explains the 2 mg/dL error and the biases.

So I'll rebuild `with_code` explicitly in the A2 form:

q + code embedding + last_numeric(A2·[1, has, age, z_last]) + last-value-bin embedding + rate-bin-0 embedding + last_status term.

I'll first find the register features (has, age, z_last, last bin, code index) in `M4`'s columns. If my rebuilt means equal `mA2`, I'll have trustworthy `with_code` vectors for the probe and the cost checks. If they don't, the last_status or rate-bin term is placed differently than I assume, and I'll test the variants.

```python
print(M4.columns.tolist())
print(M4.iloc[:3].T.to_string()[:1500])
print('mA2 vs M4.predicted_mean?', np.abs(mA2-M4['predicted_mean'].values).mean() if 'predicted_mean' in M4 else '')
print('last_status rows norm', np.linalg.norm(L('last_status.weight'),axis=1), ' rate bin0 norm', np.linalg.norm(L('embed.bins.rate.weight')[0]), ' value bin0 norm', np.linalg.norm(L('embed.bins.value.weight')[0]))
print(VOCAB['codes'] if isinstance(VOCAB['codes'],list) else list(VOCAB['codes'].items())[:13])
```