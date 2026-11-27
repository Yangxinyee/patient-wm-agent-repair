The current edit (dims 49 and 185 set aside, nothing written into them yet) cost about +0.0034 nats on 100 tuning stays. That fits within the 0.005 allowance but uses most of it. Next I'll record this in the notebook and re-read the parts of the architecture covering the input path (`delta_in`, flow, `b_in`, `z_post`, prelude) so I can plan the write-in path.

```python
with open('/work/notes.md', 'a') as fh:
    fh.write("\n## Session 8\n- Current edit = free dims 49,185 (zero writes in prelude/blocks/coda, zero norm w/b, c_out/b_in row+col). fwd_6 (100 tuning): NLL 1.52794 -> 1.53133 (+0.0034). No injection/readout yet.\n")
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
import re
for i, line in enumerate(txt.splitlines()):
    if re.search(r'delta|flow|b_in|a_in|z_post|z_in|prelude|token', line):
        print(i, line[:160])
```