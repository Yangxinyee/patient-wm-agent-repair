The head helpers are row-aligned arrays over the 5,673 factual readings:
- `V0`: the head's view without z;
- `fixed`: the code-specific additions;
- `TRr`: transition logits;
- `cen`: bin centres.

`probs_from_v0` adds the z term from the prior and runs the value head.

To score a perturbed state I need to rebuild V0 from a state (out, y, time gap) and know which bundle each row belongs to. Next I will:
- look at one `H` entry;
- write `view0(out, y, dt)`, following the notebook formula;
- check it against stored `V0` rows using the states from `run_tf`.

A near-zero error lets me compute exactly how the next predicted glucose responds to a change in the state alone.

```python
print(type(H[0]), str(H[0])[:300])
def view0(out,y,dt):
    return out+ln_((y*torch.exp(-dt*torch.exp(W64['core.flow.a']))).mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt,1e-9))/10],dtype=torch.float64),'gap')
Rk=run_tf(B)
rows=[i for i,hh in enumerate(H) if (hh.get('sid',hh.get('stay_id')) if isinstance(hh,dict) else hh[0])==sid0][:8]
print('rows',rows)
```