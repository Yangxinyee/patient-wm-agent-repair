The edit is saved. The gates work: the glucose unit's pre-activation is at least 40 on glucose rows and at most −8 on MAP rows. The replica estimates a factual cost of 0.027, which is too high if the tool agrees. The tool test is decisive, and I need the setA stay ids for the request. I will print them and log this session's progress in the notebook.

```python
import json
notes='''
## Session 7 (runs 249-265)
- DEC ridge decoders (samp_pert) DO NOT respond to replica edits on Xf (resp 0.09 vs dose chg 1.42): invalid representation (gen_samples != Xf, offset ~9). Xf+W1.. replica head exact vs nll0.
- MVDR decoders from EDR dX (unit mean response per dd123) + within-cell cov of Rf (ridge 3.6e-5): noise sd ins 1.24, NE 0.20 (per unit). Stored DEC2[('ins','mvdr',-2.8)], DEC2[('ne','mvdr',3)].
- Replica exact dNLL of tilt edit (all rows): ins c=-2.8 0.0067, NE c=3 0.0199 (replica sharper than tool: nll 2.04 vs 1.52). Resid-informed decoders worse.
- EDIT v1 SAVED (run 264): units 0 (glucose, gate gw) & 114 (MAP), pre=B*gate+k*(v.x-m), W2 col = c/medvar*centred centres/k, TR per-cell compensation. cI=-2.8 cN=3.0. replica dNLL .0268.
- Next: tool factual both + dose_edit both on setA.
'''
open('/work/notes.md','a').write(notes)
print(json.dumps([int(s) for s in setA]))
```