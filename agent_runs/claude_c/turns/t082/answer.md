The last run only listed the saved forward passes (1–4) and the original model's factual fit on tuning stays 0–99: 1.63461 nats per reading over 4,751 readings. That number is the baseline any edit must stay within 0.005 of.

My notebook says three edits have failed so far:
- Scaling the dose and rate columns of `embed.numeric` changed the readouts only slightly (insulin x2 went from −1.85 to −2.5, against a true effect of about −13).
- Amplifying the dose and rate bin embeddings got the sign wrong.
- Amplifying the drug-bundle z26 deviation changed nothing.

So the dose signal must reach the glucose and MAP predictions through some other route. I no longer have the architecture details in context, so I need to re-read `/model/ARCHITECTURE.md` and check which kernel variables are still alive before choosing the next lever. This run reads only; no hypothesis is tested yet. It should show me every tensor through which dose or rate tokens affect later predictions (embedding, attention or slot memory, decoder).

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print('-----')
print([k for k in globals().keys() if not k.startswith('_')][:200])
```