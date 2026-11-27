Units 114 and 0 of the value head are effectively dead: their maximum activation is about 0.003 and their contribution is zero. Repurposing them for the insulin and NE corrections costs nothing factually. The bin centres differ by code (finger 73…, lab 69…, arterial 33…, cuff 37…), so the logit tilt must be designed per code group. The comp terms in embed_transition can absorb code-specific constants.

To compute the compensation and check the gating for glucose_lab and map_cuff queries, I need head inputs for those codes too. gen_samples produces them only for finger and arterial, so I need the exact format of the register argument it passes to wc_vec. This run disassembles the part of gen_samples that builds 'last' and calls wc_vec. It prints only the relevant instructions.

```python
import dis, io
buf = io.StringIO(); dis.dis(gen_samples, file=buf); L = buf.getvalue().splitlines()
idx_wc = [i for i,l in enumerate(L) if 'wc_vec' in l]
print('wc_vec refs', idx_wc[:3])
i0 = idx_wc[0] if idx_wc else 0
for l in L[max(0,i0-40):i0+12]:
    s = l.strip()
    if any(t in s for t in ['LOAD','STORE','BUILD','CALL','BINARY','SUBSCR','COMPARE']): print(s[-60:])
```