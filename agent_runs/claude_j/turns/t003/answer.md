The vocab confirms that insulin rate and dose share the same median (1.5), IQR (1.0) and fine-bin edges. Norepinephrine rates are capped at 1.0 mcg/kg/min, since the top edges repeat. The fine-bin tables `embed.bins.dose` and `embed.bins.rate` are shared across codes (17 x 192), and so is `embed.numeric`. So an edit at the embedding level cannot single out insulin unless I exploit the code embedding. The cleaner place to intervene is likely inside `encode_action` or downstream. Next I will list the remaining tensor names and load the training records to see their structure: event frequencies, dose grids and timing. This tells me whether doses look protocol-driven, for example a deterministic function of the last glucose, or random. That decides how the causal effects can be identified.

```python
names = list(T.keys())
rest = [n for n in names if not n.startswith('core.blocks') and not n.startswith('core.prelude') and not n.startswith('embed.') and not n.startswith('encode.')]
for n in rest: print(n, T[n])
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.dtypes)
```