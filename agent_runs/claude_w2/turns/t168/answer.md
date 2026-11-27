The last run looked at where the prelude attention sends information on insulin bundles. Heads 0 and 2 attend entirely to the drug-action token u, and heads 1 and 3 mostly to u. So the dose reaches the recurrent state only through u, at the bundle where the drug starts. The memory holds e, which carries no rate or dose. That means no ready-made "dose history" channel exists that the value head could read.

My working plan is a correction in the value head: a gated hidden unit that shifts the glucose (or MAP) prediction by a decoded "dose deviation" term. The term is fitted to the recorded data and to counterfactual teacher-forced states that I compute with my exact local re-implementation. To build it I first need the APIs of my own code files, since kernel functions have no visible source. Next I print the function signatures and the full rollout and head code.

```python
import re
for f in ['impl.py','core_test.py','head.py','rollout.py','tf.py']:
    p='/work/'+f; s=open(p).read()
    print('=====',f,len(s),'chars',s.count('\n'),'lines')
    for i,l in enumerate(s.split('\n')):
        if l.startswith('def ') or l.startswith('class ') or (re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=',l) and len(l)<120):
            print(f'{i:4d}: {l[:150]}')
print('===== rollout.py full')
print(open('/work/rollout.py').read()[:7000])
```