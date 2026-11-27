Run 226 settles the question about drug-specific handles. In the rollout only glucose and MAP values are drawn; every other event, including the edited drug records, is taken from the record. The heads read each reading's own code row and its own register of last values. So drug information reaches glucose and MAP predictions only through the action token u_k, which is normalised twice before the prelude attention.

The insulin and norepinephrine code rows are the only drug-specific handles. They are constants, though, so editing them can only rotate the normalised token. That cannot raise the downstream response, which the u=0 test showed is small (mean |shift| 1.1 for glucose, 0.3 for MAP).

I therefore conclude that no selective lever remains beyond input-side amplification. Within that family, N5 (x5) beat x20 and x1 on the readouts. The alternatives with extra gain all cost too much factual fit: attention-head boosts reach +0.0037 to +0.0051 against the 0.005 limit.

My decision is to keep N5 as the final edit. Before answering I'll record this in the notebook and confirm that the two stored operation vectors are the z_dose and z_rate unit vectors. That check confirms the submitted ops are exactly the tested N5 edit; any other vectors would mean the tested numbers don't apply.

```python
import numpy as np
print(np.load('/work/vectors/v001.npy').round(3), np.load('/work/vectors/v002.npy').round(3))
open('/work/notes.md','a').write("""
## Restart-12 (runs 219-227)
- Pass 31 = N5 factual on tuning 200-299: 1.61959->1.62151 (+0.0019; gf +.0046, gl +.0133, art -.0033, cuff 0).
  Pooled 300 tuning stays: +0.00212 +- 0.00081 (run 218).
- Re-read ARCHITECTURE: drug info enters ONLY via u_k (DeepSets+LN) -> prelude attention (LN1) -> prelude_norm -> E.
  Heads use own-code register; rollout samples only glucose/MAP values (other events as recorded).
  Drug-specific handles = constant code/unit rows only -> can only rotate u, not raise potency. No new lever.
- FINAL DECISION: submit N5 (ops 1-2). Expected readouts ~ ins x2 -2.4, half +1.75, ne x2 +0.15, ne half -0.1.
""")
print(ops())
```