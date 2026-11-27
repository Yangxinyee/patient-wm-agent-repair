The inventory shows the kernel kept its working tools: `kill`/`kill_dims`, `make_scaffold`/`scaffold_v`, the offline replicas (`tf_nll`, `tf_eval`, `tf_loop`, `rollout_stay`, `rollout_gen`), `true_shift`, and the target helpers. So I do not need to rebuild the simulator.

Two tensor groups are missing from my notebook's architecture recap: `filter_gate`/`filter_norm` and `pending_mlp`. If one of them carries running-infusion information into the head or the state, my belief that the prelude is the only dose route is wrong, and it would offer a simpler place to edit.

Next I will print the signatures of the key kernel functions, so I can call them correctly without their sources. I will also grep ARCHITECTURE.md for "pending" and "filter", with short excerpts only. If "pending" turns out to feed drug rates into the value head's view, I will consider editing there instead of building the memory circuit.

```python
import inspect, re
for f in ['kill','kill_dims','make_scaffold','scaffold_v','rollout_stay','rollout_gen','true_shift','tf_nll','tf_eval','tf_loop','head_value','build_ds','prelude','core_rest','mem_for','cost','score','report','fit_eval','wfit_eval','deconv','lag1_rate','clone_P']:
    try: print(f, str(inspect.signature(globals()[f]))[:150])
    except Exception as ex: print(f, 'ERR', ex)
A=open('/model/ARCHITECTURE.md').read()
for kw in ['pending','filter']:
    for m in re.finditer(kw, A):
        print(kw, '|', A[max(0,m.start()-150):m.start()+200].replace('\n',' ')); break
print('ARCH len', len(A), 'n pending', A.count('pending'), 'n filter', A.count('filter'))
```