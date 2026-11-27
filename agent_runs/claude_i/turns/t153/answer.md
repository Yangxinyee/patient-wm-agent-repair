The last run printed ARCHITECTURE.md. Dose information reaches later glucose and MAP predictions only through the action-encoder output u_k. From there it enters the state: through the prelude tokens and z_post, then the recurrent blocks and the coda. The "last" register of the value head only holds the same code's previous value, so it carries no dose information for glucose or MAP. Drug END atoms also carry no dose.

So a correct causal response has to be produced by how u_k changes the state h/out and how the heads read that state. I want to test edits locally instead of spending forward passes. For that I need a CPU reproduction of the full core step (prelude, blocks, coda), checked against the recorded states in fwd_3.

Next I print my helper files (enc.py, head.py), the helper functions defined in the kernel, and the core/z/encode_action tensor shapes. This is inspection only. It will confirm whether a core-step reproduction already exists or must be written, and whether the coda/c_out structure is residual enough to hold a persistent dose-memory channel.

```python
print(open('/work/enc.py').read())
print("=====")
print(open('/work/head.py').read())
print("=====")
fn=[k for k,v in globals().items() if not k.startswith('_') and type(v).__name__=='function']
print(fn)
print("=====")
for nme in NAMES:
    if nme.startswith('core') or nme.startswith('z_') or nme.startswith('encode_action') :
        print(nme, SHAPES[nme])
```