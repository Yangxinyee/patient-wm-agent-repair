Run 36 compared three orderings of the blocks of a weight matrix, probably the gates of a recurrent cell in my own re-implementation of the model. The identity order (0,1,2) with no rescaling fit best (error 0.091). That suggests the re-implementation reads the blocks in their stored order.

The conversation was restarted, so I no longer have the architecture details or what runs 28–36 did. Before editing anything I will re-read ARCHITECTURE.md and list my kernel variables and the forward-pass outputs. That way I don't rebuild what already exists.

My working hypothesis, from the notebook:
- **What the records show:** insulin lowers glucose by about 2 mg/dL per U/h within the first hour, with a total effect of about 13.7 mg/dL per U/h spread over about 10 hours. Norepinephrine raises MAP by about 3.6 mmHg per mcg/kg/min in the first hour, about 10.4 in total.
- **What the model shows:** it gives only about a quarter of these effects in the dose questions, even when it reads the real record.

Plan: strengthen the dose pathway (the dose and rate embeddings, then the action encoding) so the effect in the model's predictions matches the records. Then check with the forward tool that factual NLL does not get worse. The notebook says it should improve, because the model under-uses doses even when it reads the real record.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print(os.listdir('/work'), {d: os.listdir('/work/'+d) for d in ['fwd_1','fwd_2','fwd_3']})
g = {k: (type(v).__name__, getattr(v,'shape',None)) for k, v in globals().items()
     if not k.startswith('_') and k not in ('In','Out','exit','quit','get_ipython') and not callable(v) and type(v).__name__!='module'}
print(g)
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and type(v).__name__=='function'])
```