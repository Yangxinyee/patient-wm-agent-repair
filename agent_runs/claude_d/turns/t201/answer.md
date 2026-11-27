E2 is saved as the current edit: four value-head tensors, matching my arrays to float32 precision. The script then failed only because sel1.json is a dictionary, not a list, so the notebook entry was not written. I now inspect its structure to get the 100 tuning stay ids for the dose-edit request, and write the notebook entry.

```python
sel1 = json.load(open('/work/sel1.json'))
print(type(sel1).__name__, list(sel1.keys())[:5] if isinstance(sel1, dict) else None)
k_ = list(sel1.keys())[0]; v_ = sel1[k_]
print(k_, type(v_).__name__, (len(v_) if hasattr(v_,'__len__') else v_), (v_[:3] if isinstance(v_, list) else None))
with open('/work/notes.md','a') as f:
    f.write("""
- Runs 166-180: gated-channel head edit. Repurpose hidden units (31,143)=insulin, (54,9)=NE (joint ablation dNLL +.0003, 4-draw).
  unit a: M*(AND gate - th) + s/lam, unit b: M*(AND gate - th); W2 cols +-lam*t (exp tilt). AND gate = family gate + presence(drug at t-1) gate (ridge on z-aug wc).
  Decoders on present rows (z-aug ridge): ins eps1 R2 .59 k .57, eps2 .24; NE D1 .19 k .25. Present-row deficits: ins -1.6,-1.33,-0.39; NE joint 3.15,.53,-2.13.
  E2 (a_ins=(-2.8,-5.5), a_ne=8.7): offline dNLL +.0063 (gf +.0117) over budget; resid slopes fixed. SAVED as current edit for dose_edit test (fwd 8).
""")
```