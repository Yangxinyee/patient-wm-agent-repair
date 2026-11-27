The conversation restarted. Run 146 probed how a perturbation at the end of the prelude passes into `u`, and I no longer have the context for those numbers. Runs 127–146 are also missing from my notebook, and the replies after run 146 were blocked.

So before going further I will re-orient cheaply. I will list the functions and large variables in the kernel and print only the section headers of ARCHITECTURE.md. I will also add a session marker to the notes. All outputs will stay short.

My working hypothesis is unchanged. The model reacts to dose changes at only about 15–25% of the true size. A correction needs to respond to the dose's departure from its expected value (close to zero on recorded data) so that factual fit is kept. The open question is how to carry that signal through the state into the value head.

This step is bookkeeping, so it cannot confirm or refute anything. It only tells me which tools I still have.

```python
import types, os
fns=sorted(k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_'))
print('fns:',' '.join(fns)[:1200])
big=[(k,type(v).__name__) for k,v in globals().items() if not k.startswith('_') and type(v).__name__ in ('dict','ndarray','Tensor','DataFrame','list')]
print('vars:',' '.join(f'{k}:{t}' for k,t in big)[:1200])
txt=open('/model/ARCHITECTURE.md').read()
print('arch chars',len(txt))
print([(i,l[:50]) for i,l in enumerate(txt.splitlines()) if l.startswith('#')][:40])
for f in ['heads.py','prep.py','sel1.json','sel_train.json']:
    print(f, os.path.getsize('/work/'+f))
with open('/work/notes.md','a') as f:
    f.write('\n## Session 6 (run 147+): restarted; runs 127-146 undocumented (last: probe end->u attrs). Re-orienting.\n')
```