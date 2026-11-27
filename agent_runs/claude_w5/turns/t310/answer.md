The last pass (fwd_9) brought the two ×2 readouts close to my record-based targets: insulin ×2 −14.6 (target −14.3) and norepinephrine ×2 +3.96 (target +4.11). The two ×½ readouts are wrong. Insulin ×½ is −2.99 (target +7.15, wrong sign) and norepinephrine ×½ is −0.42 (target −2.05).

If I split each readout into a part proportional to the dose change plus a common offset, the offset is large: about −6.9 mg/dL for insulin and +1.0 mmHg for norepinephrine. My edit responds to *any* edited dose as if extra drug had been given. Changing the kappa scale cannot fix this, because it scales the offset too.

The leading explanation is in my notes. The tool's dose_edit rollout encodes drug atoms from hour 12 differently from the factual encoding, possibly in bin convention or field handling. My rate decoders are sensitive to that difference, while the original model is not.

With only 18 Python runs left, the next run is a single combined diagnostic. It will:
- log fwd_9 to the notebook;
- list the kernel functions and variables I can reuse;
- re-read the README and the dose-edit and binning lines of ARCHITECTURE.md;
- look at the per-hour and per-stay structure of the tool's dose_edit output.

If the ×½ error comes from a few stays or a specific hour pattern, it is an encoding artifact I can target. If the ×½ shift is near zero at h13 for most stays and grows uniformly, the artifact acts at every edited bundle.

```python
import inspect, re, numpy as np, pandas as pd
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 7 (runs 283+)\n"
            "- fwd_9 (dose_edit edited selA, c_I=1.5, c_N=1.3, hash a56a7f6079): ins x2 -14.60, x1/2 -2.99 (WRONG SIGN; target +7.15), "
            "ne x2 +3.96, x1/2 -0.42 (target -2.05). Fit E(s)=a(s-1)+b: ins a=-7.74 b=-6.86; ne a=2.92 b=1.04 -> common-offset artifact.\n"
            "  fwd_6 h13 (c=1): tool-local artifact -0.42 (x2), -1.19 (x1/2); recorded-run drift edited-orig grows ~linearly (const per-bundle bias?).\n")
G = dict(globals())
fn = []
for k, v in G.items():
    if k.startswith('_'): continue
    if inspect.isfunction(v) and getattr(v, '__module__', '') == '__main__':
        try: fn.append(f"{k}{inspect.signature(v)}")
        except Exception: fn.append(k)
print('FUNCS:', '; '.join(sorted(fn)))
vs = []
for k, v in G.items():
    if k.startswith('_') or inspect.ismodule(v) or callable(v) or k in ('In','Out'): continue
    s = getattr(v, 'shape', None)
    if s is None:
        try: s = len(v)
        except Exception: s = ''
    vs.append(f"{k}:{type(v).__name__}{s}")
print('VARS:', ' '.join(sorted(vs)))
print(open('/data/README.md').read()[:2500])
A = open('/model/ARCHITECTURE.md').read().splitlines()
hits = [i for i, l in enumerate(A) if re.search(r'dose.?edit|scal|digitiz|searchsorted|right|edge|round|bin', l, re.I)]
for i in hits[:45]: print(i, A[i][:220])
for k in (6, 9):
    d = pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv')
    if k == 9: print(d.columns.tolist()); print(d.head(3).to_string())
    cols = d.columns.tolist()
    ec = [c for c in cols if 'edit' in c.lower()]; hc = [c for c in cols if c.lower() in ('hour','stay_hour','h')] or [c for c in cols if 'hour' in c.lower()]
    sc = [c for c in cols if 'shift' in c.lower()]
    try:
        if ec:
            t = d.groupby([ec[0], hc[0]])[sc[0]].mean().unstack(0).round(2)
        else:
            t = d.groupby(hc[0])[sc].mean().round(2)
        print(f'fwd_{k} per-hour mean shift:\n', t.to_string())
        if ec:
            p = d[d[hc[0]] == d[hc[0]].min()].pivot_table(index='stay_id', columns=ec[0], values=sc[0])
            print(f'fwd_{k} first-hour per-stay quantiles:\n', p.quantile([.1,.25,.5,.75,.9]).round(2).to_string())
            print('corr first-hour:\n', p.corr().round(2).to_string())
    except Exception as e:
        print('analysis failed', k, repr(e))
```