The last pass showed the edit failing in the rollout. The original model gives insulin x2 −0.55 and NE x2 +0.19. The edited model gives +0.37 and +0.05, so for both drugs the edit's contribution has the wrong sign. In teacher-forced local checks the edit moved the next reading the right way. So something in the rollout makes the decoded "dose residual" respond in the opposite direction. My guess is that c_hat or the decoder picks up the previous, also-scaled doses carried in the state.

Before changing anything, I need two things. First, the early notebook entries: how the true effects (g-estimation targets) were estimated, and the data structure. Second, the per-hour breakdown of the rollout shifts. If the edited shift is negative right after hour 12 and then reverses, that supports the c_hat/previous-dose explanation. If it is wrong from the first hour, the problem is in the gate or decoder.

```python
txt = open('/work/notes.md').read()
print(len(txt))
idx = txt.find('## Runs 70-91')
print(txt[:min(idx, 14000)])
import pandas as pd, numpy as np
de_o = pd.read_csv('/work/fwd_4/dose_edit_original.csv')
de_e = pd.read_csv('/work/fwd_4/dose_edit_edited.csv')
print(de_o.columns.tolist(), de_o.shape)
print(de_o.head(3).to_string())
```