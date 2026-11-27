The targets are saved.

Next I need to know whether I had already started a local reimplementation of the encoders and prelude in the previous session (runs 47–51). The edit plan needs one to obtain E, the prelude output that `core.b_in` reads. I will list only the names of functions and torch modules in the kernel. Seeing names like an encoder or a prelude step would confirm partial work I can reuse. Otherwise I will start one.

```python
fn = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in (None,'__main__')]
print(fn[-40:])
print([k for k,v in globals().items() if isinstance(v, torch.nn.Module)][:20])
```