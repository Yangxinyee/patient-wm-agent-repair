The last output (run 207) compared ridge-fitted weight edits at four penalties (10, 50, 200, 1000). For each it showed the teacher-forced counterfactual shift: S is the true shift, M is the original model and A is the edited model. With the weakest penalty (10), the edited shift for insulin x2 from hour 12 reaches -8.3 against a true -13.5. The original model gives -1.7. So a gated head-unit edit moves the model a long way toward the truth, though not all the way. I do not remember exactly what "expF" measures, so I cannot yet tell whether the edit costs factual fit.

The kernel holds the code from the last few runs, and parts of it may not be saved to files. Next I will take a short inventory: the kernel's function and variable names, and the `def` names in /work/*.py. This will show which pieces of the edit pipeline are reusable. Success means I find the function behind "wf" and the ridge or edit-building code. If I don't, I will rebuild that step into a file.

```python
import os, re, types
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print('FUNCS',len(fn),' '.join(sorted(fn))[:1500])
big=[k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,(types.FunctionType,types.ModuleType,type)) and k not in ('In','Out')]
print('VARS',len(big),' '.join(sorted(big))[:1500])
for f in ['cf.py','gen.py','heads.py','prep.py']:
    s=open('/work/'+f).read()
    print(f,len(s),'defs:',' '.join(re.findall(r'^def (\w+)',s,re.M))[:300])
```