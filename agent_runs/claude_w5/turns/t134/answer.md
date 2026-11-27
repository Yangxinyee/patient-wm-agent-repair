The tensor list confirms the dimensions:
- **Value head:** a 2-layer MLP, 192 → 192 hidden → 17.
- **z_post:** an MLP, 576 → 192 hidden → 64, of which only the first 32 (the mean) are used.
- **FFNs:** the prelude, block and coda FFNs, and encode_action, each have 768 hidden units.
- **Linear maps:** b_in and c_out are bias-free 192×192 maps.
- **last_numeric:** takes 8 inputs, not the 4 the documentation lists.

My kernel already has a head implementation (`head_probs`, `head_inputs`, `cmp_head`) and a `free_run`. These come from before the restart and I don't remember their interfaces. Next I will print the rest of the function list with signatures, plus the fwd_3 summary (a factual pass of the original model). Then I can reuse the head code and check it against the tool's predicted means rather than rebuild it.

```python
import inspect, json
fns = sorted([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__'])
for k in fns:
    try: sig = str(inspect.signature(globals()[k]))
    except Exception as ex: sig = '?'
    print(k+sig[:90], end=' | ')
print()
print(json.dumps(json.load(open('/work/fwd_3/summary.json')))[:700])
print(json.dumps(json.load(open('/work/fwd_2/summary.json')))[:400])
```