The attention pattern explains why v1 failed. At drug bundles, prelude heads 1–3 put about 99% of the slots' attention on the u_k token, and head 0 attends mostly to the slots. The z token gets essentially zero weight in every head, so no z_post edit can reach the state.

The working carrier is therefore the u_k token's value path, or prelude FFN units that read the slots after attention. To design a gate that fires only at drug bundles, I first need the attention pattern at non-drug (reading) bundles. That requires e_k there, which I only replicated approximately before. I'll check my existing helper functions to rebuild it.

```python
import inspect
for f in ['e_from_atoms_batch','meas_atom','meas_parts','enc_e','set_enc','e_drug_bundle','u_bundle','ctx_of']:
    try:
        src = inspect.getsource(globals()[f]).split('\n')
        print(f, '|', src[0][:150], '| lines', len(src))
    except Exception as ex: print(f, 'err', ex)
```