The NLL check is decisive.

| Correction | Change in NLL per reading |
|---|---|
| Raw dose (insulin on glucose) | about +0.08 |
| Raw dose (norepinephrine on MAP) | +0.06 to +0.25 |
| Residual, d − expected dose | −0.001 to −0.003 (slight improvement) |

The 0.005 budget therefore rules out simply amplifying the dose pathway. The edit has to act on the dose relative to what the record's own history predicts. In rollouts that residual grows large under doubled doses; in teacher forcing it stays small.

A structural problem remains for memory:
- In my notes `a_bar` is 0.27–0.47 per bundle, so h forgets fast.
- The memory M stores only the bundle vectors e, not the action encodings u. A 12-hour dose memory can only come from M via cross-attention.
- That works only if the infusion END atoms are encoded into e carrying the ended record's rate and dose.

I'll check what END atoms carry in my bundle builder and atom featurizer, and save the finding to notes.

```python
import inspect, re
with open('/work/notes.md','a') as f:
    f.write("- fwd_4 factual original tuning batch1: value NLL 1.55781 (n 4989). finger 1.4758, lab 1.3044, art 1.3964, cuff 1.9034.\n"
            "- KEY: model error regresses on dose RESIDUAL (finger eps1..3 -1.36,-1.14,-.62 se.7) not on policy part (pi coefs ~0) => model absorbed policy.\n"
            "  Raw-dose correction would cost dNLL ~+0.06..0.25/reading; residual-based correction improves ~-0.002. => edit must act on d - expected d.\n")
src=open('/work/setup.py').read()
i=src.find('def build_bundles'); print(src[i:i+1400])
src2=open('/work/wm.py').read()
i=src2.find('def atom_matrix'); print(src2[i:i+900])
```