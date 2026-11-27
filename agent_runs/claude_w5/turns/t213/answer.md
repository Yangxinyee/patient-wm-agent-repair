The core step returns (out, y, z, E, hf, and a sixth 4×192 tensor, probably h_new). Its recorder only exposes E and the prelude FFN hidden layer. To build and check the edit I need every intermediate: the attention inputs and probabilities, the slot residual before and after the prelude FFN, the FFN hidden units, xb per layer, h_new and y.

Next I write a transparent step function from the documented architecture in the notebook and verify it against cstep_h's stored results on all bundles of one stay. If it matches to about 1e-9, I use it for all edit design. Any mismatch tells me which documented detail I have wrong.

```python
import math
F_ = torch.nn.functional
known = ('core.a_in','core.age','core.b_in','core.blocks','core.c_out','core.coda','core.delta_in','core.flow','core.out_norm','core.prelude','core.prelude_norm')
print('other core keys', [(k, tuple(W64[k].shape)) for k in W64 if k.startswith('core.') and not k.startswith(known)])
print('gap/z keys', [(k, tuple(W64[k].shape)) for k in W64 if k.startswith(('gap','z_post','z_in','core.age'))])
```