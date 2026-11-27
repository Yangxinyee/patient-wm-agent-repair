The factual baseline is 1.549 nats per reading. Residual spread is about 15 mg/dL for glucose and 5.2 mmHg for arterial MAP. That makes the MAP side very sensitive: a systematic bias of only about 1 mmHg on arterial MAP would use up the whole 0.005 tolerance. An uncentred amplification of the dose pathway is therefore ruled out. The correction has to respond only to the part of each dose not already implied by the history.

Designing and checking such an edit safely needs an exact offline copy of the model's step. Only that lets me compute factual predictions and dose-edit rollouts locally instead of spending forward passes. I'll inspect the helpers I already wrote to see how far the re-implementation got.

```python
import inspect
for nm in ['lin','mlp','ln','LN','act','zsc','fbin','atom_vec','set_enc','first_err']:
    try:
        src=inspect.getsource(globals()[nm]); print(f'--- {nm} ({len(src)} chars)'); print(src[:700])
    except Exception as e: print(nm,'ERR',e)
```