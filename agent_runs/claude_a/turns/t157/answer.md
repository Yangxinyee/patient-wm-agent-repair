Run 152 located the cost: zeroing the h content of the 8 dims alone costs 0.144 nats. The head-side zeroing partly offsets it, which is why the full scaffold came to 0.078. Freeing the 4 FFN and 2 value-MLP units costs almost nothing (0.0002).

The reason zeroing is so expensive is that it removes the dims' within-step content and constant level everywhere: E, the prelude residual and the LayerNorm statistics all shift. My earlier dimcost kill was cheap because it only replaced the carry-over of h[j] with its mean. Every within-step use stayed intact.

So I'll repurpose by mean replacement, not zeroing. Concretely:
- Keep each dim's constant level.
- Cut a reader by setting that LayerNorm's gain to 0 and its bias to the original mean of the normalized value.
- Replace only the specific within-step writers a channel needs, and price each replacement separately.

For that I need per-LayerNorm means of the normalized values on the h-stream, which requires intermediate activations. I'll check whether my validated `step_b` returns them (`ret=True`) and how `prelude_parts` lays out its outputs. That decides whether I can collect the means without writing a new instrumented step.

```python
import inspect
for fnm in ['step_b','LN','MH','FF','prelude_parts','kill_dims','tf_states_ds']:
    co = g[fnm].__code__
    print(f'{fnm}{str(inspect.signature(g[fnm]))[:90]}\n  locals={co.co_varnames[co.co_argcount:]}\n  names={co.co_names}\n  consts={[c for c in co.co_consts if isinstance(c,(str,int,float))][:40]}')
```