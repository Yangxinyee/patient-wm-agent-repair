That result settles the head-only route: it is infeasible. The paired-perturbation test shows the head input barely moves with insulin dose, about 0.006 per u/h in a 192-dim vector of norm ~12. The best unit-response decoder therefore has a factual noise sd of ~68 mg/dL, against a target innovation term sd of 2.2.

The earlier R² of 0.70 for decoding perturbed doses came almost entirely from doses tracking glucose, not from a dose signal in the state. So any correct edit must amplify the dose signal upstream (in the core step) and centre it on the policy-expected dose. Otherwise factual predictions get biased by hundreds of times the NLL budget.

Before designing that, I need exact facts I don't have in my notes:
- how the value head produces fine-bin probabilities, which decides how a mean shift can be implemented;
- how z and readings are drawn in the dose_edit rollout.

I'll grep ARCHITECTURE.md for those passages and print short excerpts only.

```python
txt = open('/model/ARCHITECTURE.md').read()
import re
for kw in ['fine', 'rollout', 'z_prior', 'prior', 'sample', 'transition', 'value_bin', 'within']:
    hits = [m_.start() for m_ in re.finditer(kw, txt)]
    print('==', kw, len(hits))
    for h_ in hits[:2]:
        print('   ...', txt[max(0,h_-150):h_+250].replace('\n',' ')[:400])
```