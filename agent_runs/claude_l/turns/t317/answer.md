Run 289 showed that the least-squares direction is unusable. To reproduce the E-change of one unit of dose while leaving attention scores untouched, it needs an enormous norm: about 1,000 (insulin) and 3,750 (norepinephrine), against a typical token norm of about 13. Pushed through the token's layer norm, such a perturbation just replaces the token. The z-channel then loses its attention, and the rollouts come back identical to the original model at every gain.

The same run quantified why the old direction saturated. Moving the copy token along the dose direction by one unit reproduces the unit-dose effect well (9% error for insulin). At 4 units the error is 41%, and at 10 units 90%.

The underlying problem is leverage. The dose direction in token space moves predicted glucose very little: about 0.6 mg/dL per 3.6 token units. Getting the 3–4× larger effect the records imply would need token moves larger than the token itself.

A much stronger lever exists in the model already: the way a reading's value enters the state. At drug bundles, attention heads 1 and 2 read the u/copy token, and the same heads carry reading values at reading bundles. So the copy token could carry a small "reading-like" message, proportional to the dose residual, that shifts the model's belief about glucose (or MAP) directly.

Building that needs:
- the prelude attention's value and output weights;
- a direction in the encoded-observation token for "this reading is 1 unit higher";
- the bundle/atom format, to construct that direction if no saved one exists.

Some leftover arrays (`Dtr`, `Dtu`, keyed by reading type, and `Dmin_*`) may already be such directions. So this run is an inventory of exactly those pieces. If `Dtr` turns out to be per-unit reading directions with sensible norms, I can build the message direction next run without constructing atoms.

```python
L_=src_all['wm2.py'].split('\n')
for i,l in enumerate(L_):
    if l.startswith(('def bmha','def blin')): print('\n'.join(x[:170] for x in L_[i:i+7] if x.strip())); print('--')
print([ (n,tuple(SHAPES[n])) for n in NAMES if 'prelude' in n])
for nmD in ['Dtr','Dtu','DIRS','NUMV']:
    D9=globals()[nmD]; print(nmD, {str(k9):(getattr(v9,'shape',None) if not isinstance(v9,dict) else list(v9)[:6], round(float(np.linalg.norm(v9)),3) if isinstance(v9,np.ndarray) else '') for k9,v9 in list(D9.items())[:8]})
print('Dmin norms', np.linalg.norm(Dmin_ins), np.linalg.norm(Dmin_ne), 'cos Dmin_ins/VD_old', float(Dmin_ins@VD_old['ins']/np.linalg.norm(Dmin_ins)/np.linalg.norm(VD_old['ins'])))
s9=list(BUN)[0]; b9=BUN[s9]; print('BUN[s] type', type(b9).__name__, len(b9), 'bundle0 type', type(b9[0]).__name__, repr(b9[0])[:300])
rb=[x for x in b9 if 'glucose_finger' in repr(x)][:1]; print('reading bundle:', repr(rb[0])[:600] if rb else None)
for fn9,s_ in src_all.items():
    j=s_.find('def edit_atom')
    if j>=0: print(fn9, s_[j:j+400])
```