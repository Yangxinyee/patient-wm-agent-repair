Run 263 confirmed `run_cf` still works. The withheld replies most likely failed because of long output, so from now on I will print only short summaries.

Where things stand: the head-only edit v2 moved the drug-I readout only from −2.75 to −3.49, against an estimated truth near −12.9. It also costs +0.0044 factual NLL, which is too close to the 0.005 limit. The reason is that the core barely carries the dose value into its state, so the value head cannot tell that a dose was changed.

New plan: edit the core so it keeps a clean running sum of past doses, decaying over time, in a little-used state dimension. The value head will then read that sum minus its context-predicted part. First I need the exact form of the core's memory (h) update.

- **Confirms the plan:** h is updated by adding to a decayed h, with a per-dimension decay rate (flow.a).
- **Refutes it:** h is gated or fully normalised, so an added signal would not persist. In that case I would put the channel into the residual path from the previous output instead.

```python
import re
A=open('/model/ARCHITECTURE.md').read(); L=A.splitlines()
hdr=' | '.join(f"{i}:{l[:50]}" for i,l in enumerate(L) if l.startswith('#'))
out=[f"n={len(L)}", hdr[:1500]]
pat=re.compile(r'slot|flow|\bh\b|h_new|write|decay|residual|out_p|out =|out=', re.I)
m=[f"{i}:{l.strip()[:130]}" for i,l in enumerate(L) if pat.search(l)]
out.append('\n'.join(m[:28]))
s='\n'.join(out); print(s[:3800])
```