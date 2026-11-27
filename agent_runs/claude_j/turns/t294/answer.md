The rollout barely moved:
- insulin ×2: −2.75 → −3.49, and ×1/2: +1.38 → +1.94;
- drug B ×2: +0.69 → +0.75, and ×1/2: −0.58 → −0.64.

Teacher-forced, the same edit should add several mg/dL per glucose reading, so the readout does not carry over to rollouts. My explanation: the cf states it was fit on combine scaled doses with recorded readings, so the state builds up a surprise signal when the readings fail to respond to the dose. The readout probably learned that surprise. In a rollout the sampled readings agree with the model, so no surprise appears.

The self-consistent fix is to fit on exact cf histories: doses scaled and readings shifted by the true effect. If the edited model is right, its rollouts look like those histories. The kernel had code for that (run_cf with SHIFT, and mod_events_new), and /work holds gen.py, gen2.py, cf.py and amp.py.

This run lists the functions defined in those files and the signatures of the key kernel functions, so I can reuse the generator. I also log the result in the notes.

```python
import re, inspect
for fn_ in ['gen.py','gen2.py','cf.py','amp.py','prep.py']:
    p='/work/'+fn_; s=open(p).read(); print(fn_, len(s), [m for m in re.findall(r'^def (\w+)\(([^)]*)\)', s, re.M)][:12])
for f_ in [run_cf, mod_events_new, true_shift, shifted, shifts, cf_stay, gen_stay, gen2, amp_eval, rows_truth, heads_on_states]:
    co=f_.__code__; print(f_.__name__, co.co_varnames[:co.co_argcount], f_.__defaults__)
open('/work/notes.md','a').write("- fwd5 factual both sel1: orig 1.51162 -> v2 1.51604 (+.0044; pred in-sample .0039). by event dNLL: GF +.0034, GL +.0247(!), MA -.0007, MC +.0039.\n- fwd6 dose_edit both sel1, v2 g1.5: ins x2 -2.75->-3.49, half 1.38->1.94, NE x2 .69->.75, half -.58->-.64. => readout fit on cf (doses scaled, readings recorded) does NOT transfer to rollouts (likely reads 'surprise'). Need exact-cf (readings shifted by true S) training states.\n")
```