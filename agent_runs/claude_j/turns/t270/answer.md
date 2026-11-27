Run 242 changes the picture. The model's responses add up. Lowering the readings by 13 lowers its glucose prediction by 11.1, doubling the doses lowers it by 0.9, and doing both lowers it by 12.1. So the model follows its own readings strongly (persistence about 0.85), and there is no hidden interaction cancelling them.

That means my stored counterfactual dataset was built wrongly. Its "shifted" readings never reached the replica's state, which is why the model seemed to respond with only 13% of the truth.

The real deficit is the model's direct, one-step response to dose, which is about −0.9 per step where roughly −2 to −3 is needed. With persistence near 0.85, the rollout compounds whatever direct effect the model has. So a head edit only needs to add about −1.5 mg/dL per step of direct effect. At the measured signal-to-noise ratio of 1.4, that should cost only about 0.001 nats of factual fit, well within budget.

Next I will record this in the notebook, then disassemble `amp_eval` in compact form. I need its exact `run_cf` and `heads_full` calls so I can write a version that also returns the head features for insulin and norepinephrine counterfactuals.

```python
open('/work/notes.md','a').write('''
## KEY (run 240-242): M cf data BUG - shifted readings never reached replica state (obs col shifted, states not).
- Replica test (30 stays): readings shifted -13 from h13 (core+head last_val; RT age is SECONDS) -> GF pred -11.1; doses x2 alone -0.90; both -12.07 (additive).
- => model persistence ~0.85; problem = direct (one-step) dose effect too small (-0.9/step vs needed ~-2..-3).
  Needed per-step direct kernel K_D(k)=KI(k)-0.85*KI(k-1): [-2,-0.7,-0.12,+0.1,+0.17..] per unit; sum -1.9. Edit needed is SMALL (~-1.5 mg/dL/step).
- Plan: Fisher readout w from WC (cf doses x2 readings recorded vs factual), head unit tilt with gain g; calibrate g via tool dose_edit.
''')
import dis, io
buf=io.StringIO(); dis.dis(amp_eval,file=buf); L_=[l.split() for l in buf.getvalue().split('\n') if ('LOAD' in l or 'CALL' in l or 'STORE' in l)]
print(' '.join((x[-1] if x else '') for x in L_)[:1500])
```