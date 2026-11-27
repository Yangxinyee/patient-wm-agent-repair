The architecture confirms the NE path. NE atoms are actions and go through `encode_action` (an mlp1 → sum/√n → mlp2 → norm DeepSet), giving u_k. u_k reaches the state through the z posterior and the prelude token. Under x2, doubled rates above 1.0 fall into the shared bin 16 and get z-scores far outside training.

Two explanations for the weak x2 response (+1.03 vs +4.85) fit this:
- **(a)** The out-of-range encoding does not register as "NE present", so my head channel's presence gate never fires.
- **(b)** The gate fires, but the decoded dose saturates.

Both point to editing `encode_action.mlp1` for NE atoms in bin 16, which never occur in the records, so the edit costs no factual fit. The design differs between them, though. First I will check the mlp1 structure (whether a LayerNorm sits before it), the signatures of my helpers, and the columns of fwd11's per-stay-hour output. That output lets me test (a) versus (b) next by comparing hours after in-range versus out-of-range doubled NE atoms.

```python
import pandas as pd
print({n:SHAPES[n] for n in NAMES if n.startswith('encode_action')})
for f in ['atom_vec','set_enc','uenc','zpo','avec','avec3','true_shift_ne','ctrl_ne','build_ne2','wc_from','head_mean2']:
    c=globals()[f].__code__; print(f, c.co_varnames[:c.co_argcount], globals()[f].__defaults__)
D11=pd.read_csv('/work/fwd_11/dose_edit_edited.csv'); print(D11.shape, list(D11.columns)); print(D11.head(3).to_string())
```