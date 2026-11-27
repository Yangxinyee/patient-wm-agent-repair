The last run exceeded the 900-second limit: twelve fits with a numerical Jacobian were too slow. The kernel restarted and all variables are gone. Only /work/wm.py (the encoder, attention and core-step code) and the notebook survived.

The key finding from before the restart: even with a free context vector, z at k=1 could not be fitted. So my measurement encodings are probably wrong, or z_post's input order differs from my assumption. One likely culprit is the exact definition of the robust z-score or of the fine bins. `vocab.json` has two notes on these that I have not reread.

Next I check that the predefined helpers survived the restart, read the vocab notes, and record the restart in the notebook. If the notes define z or the bins differently from what I implemented (a scaled IQR, a different clip, or right-closed bins), that is the encoder bug.

```python
import json, os, numpy as np
print([n for n in ['NAMES','SHAPES','VOCAB','load','load_original','save_edit','drop_edit','edited'] if n in globals()])
V=json.load(open('/model/vocab.json'))
print('numeric_note:', V.get('numeric_note'))
print('value_bin_note:', V.get('value_bin_note'))
print('numeric keys sample:', list(V['numeric'].items())[:2] if isinstance(V['numeric'],dict) else V['numeric'][:2])
print('codes:', V['codes'] if len(str(V['codes']))<800 else str(V['codes'])[:800])
print('event_kinds', V['event_kinds'], 'atom_kinds', V['atom_kinds'])
print('centres keys', list(V['value_bin_centres'].keys())[:10] if isinstance(V['value_bin_centres'],dict) else len(V['value_bin_centres']))
with open('/work/notes.md','a') as f:
    f.write('\n## Restart 2 (runs 67-93)\n- Kernel died at run 93 (900s timeout; least_squares too slow). /work/wm.py has embed/set_enc/mha/core_step code.\n- Model rollout IRF (fwd_1, regression of shift on recorded doses): insulin ~ -0.4/U/h flat lags1-10 (sum -3.2) vs true -2.06,-2.45,...; NE +0.35,+0.17,.. (sum 1.4) vs true 3.3,2.65,...\n- Replica NOT exact: z mismatch even at k=1 with free ctx offset fit (rms .07) -> encoder (e) or z_post input order wrong. states: x[:,0:192]=out, h slots, z; out==out_norm(mean h) verified; row0 identical across stays; abar .27-.47.\n')
```