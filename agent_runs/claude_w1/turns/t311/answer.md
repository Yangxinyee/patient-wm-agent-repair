The last output was blocked by the content filter, so I don't know what it showed. The notebook says EDIT v1 was saved in run 264, and the next planned step was a tool check (factual "both" and dose_edit "both" on setA). Forward passes used are 6, so I can't tell whether that check already ran. Before spending more passes, I'll take stock with one short inspection:
- what each /work/fwd_k directory holds, with short summaries;
- the notebook lines about tool readouts and the true effects;
- the edited tensors.

If fwd_5 or fwd_6 hold "both" results for setA, I can judge EDIT v1 from them without a new pass. If not, the next step is that check.

```python
import os, glob, json
import pandas as pd, numpy as np
for d in sorted(glob.glob('/work/fwd_*')):
    fs = sorted(os.listdir(d))
    print(d, fs)
    for f in fs:
        p = os.path.join(d, f)
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(p)
                print('   ', f, df.shape, list(df.columns)[:10])
                if f.startswith('dose_edit'):
                    num = df.select_dtypes('number')
                    print('      means:', {c: round(float(num[c].mean()), 3) for c in num.columns[:10]})
                if f.startswith('factual'):
                    if 'nll' in ' '.join(df.columns).lower():
                        cs = [c for c in df.columns if 'nll' in c.lower()]
                        print('      nll means:', {c: round(float(df[c].mean()), 4) for c in cs})
            elif f.endswith('.json') or f.endswith('.txt'):
                print('   ', f, open(p).read()[:400].replace('\n', ' '))
        except Exception as ex:
            print('   ', f, 'ERR', str(ex)[:100])
txt = open('/work/notes.md').read()
print('notes chars', len(txt))
keys = ['fwd_', 'dose_edit', 'readout', 'truth', 'true eff', 'TRUE', 'orig model', 'original model']
for i, l in enumerate(txt.splitlines()):
    if any(k in l for k in keys):
        print(i, l[:220])
print('edited:', edited())
for n in edited():
    a = load(n); b = load_original(n)
    print(n, tuple(a.shape), 'max|diff| %.4g' % float(np.abs(np.asarray(a) - np.asarray(b)).max()))
```