The last run only listed the kernel's variables. Runs 208–222 were never written into the notebook, so I no longer know exactly what each helper function does. I also don't know what my blocked reply contained, so I am starting again from the inventory.

**What I conclude:** before building Plan A, I need a reliable list of my tools. Plan A is: the prelude attention head that reads the dose token feeds a cleaner path into the state, and the value head reads that path. The helpers I need are the local core step (step_at), the multi-step run (run_from), the with_code builder (wc_row), the prelude head outputs (prel_heads) and the read-fitting functions (fit_psi, fitread). I also need the tensor names grouped by module.

**What I will do:** list every kernel function with its arguments and size, list the tensor names by module, and show what the `cands` dict holds. I will also add a session header to the notebook.

**What would confirm or refute the plan:**
- If these local tools exist, I resume Plan A, starting with the prelude-stage read.
- If key pieces are missing, especially a value-head mean function, I write a fresh, documented module to /work so it survives restarts.

```python
import types, collections
fns = sorted(k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
lines=[]
for k in fns:
    try:
        c = globals()[k].__code__
        args = ','.join(c.co_varnames[:c.co_argcount])
        s0 = c.co_consts[0] if (c.co_consts and isinstance(c.co_consts[0], str)) else ''
        lines.append(f"{k}({args}) [{len(c.co_code)}] {s0[:50]!r}" if s0 else f"{k}({args}) [{len(c.co_code)}]")
    except Exception as e:
        lines.append(f"{k}: {e}")
print(len(fns)); print('\n'.join(lines))
print('IPython hist:', any(n in globals() for n in ('In','_ih')))
groups = collections.defaultdict(list)
for n in NAMES: groups[n.split('.')[0]].append(n)
for g,v in groups.items(): print(g, len(v), [x[len(g)+1:] for x in v][:30])
try:
    print(type(cands), {k:getattr(v,'shape',type(v).__name__) for k,v in cands.items()})
except Exception as e: print('cands', e)
with open('/work/notes.md','a') as f:
    f.write("\n## Session 10 (restart 10)\n- Run 224: inventory of kernel functions/tensors. Keep replies brief and technical; note every few runs.\n")
```