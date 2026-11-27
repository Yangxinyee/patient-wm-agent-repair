I found a concrete mismatch: the generic decoder used query type 0, while the trusted reading decoder uses type 1. I’ll correct that and compare both calculations on identical inputs, then check the retained GPU predictions. Numerical agreement would resolve the helper mismatch; a remaining GPU discrepancy would need further investigation.

```python
import numpy as np, torch

def readout125(h, out, src=None, dt=None, static=None,
               transition=None, centres=None):
    defaults125 = readout121.__defaults__
    src = defaults125[0] if src is None else src
    dt = defaults125[1] if dt is None else dt
    static = static_all122 if static is None else static
    transition = defaults125[3] if transition is None else transition
    centres = defaults125[4] if centres is None else centres
    flowed = h[src] * torch.exp(
        -dt[:, :, None] * torch.exp(W0["core.flow.a"])[None, None, :])
    ctx = (out[src] + norm0(flowed.mean(dim=1), "core.out_norm")
           + mlp0(torch.log(dt.clamp_min(1e-6)) / 10.0, "gap"))
    zmean = mlp0(ctx, "z_prior")[:, :32]
    q = norm0(ctx + linear0(zmean, "z_head")
              + W0["type_emb.weight"][1], "query_norm")
    hidden = torch.nn.functional.gelu(
        linear0(q + static, "attr_heads.value_bin.0"))
    logits = linear0(hidden, "attr_heads.value_bin.2") + transition
    prob = logits.softmax(-1)
    return (prob * centres).sum(-1), logits, prob, hidden

htest125 = state_x[:32, 192:960].reshape(32, 4, 192)
otest125 = state_x[:32, :192]
mtrust125 = next_means45(htest125, otest125)
mtest125 = readout125(htest125, otest125, src45, dt45,
                     static45, trans45, centres45)[0]
print("type-corrected / trusted max difference:",
      float((mtest125 - mtrust125).abs().max()))

mstate125 = readout125(
    state_x[:, 192:960].reshape(-1, 4, 192), state_x[:, :192],
    src=torch.arange(len(state_x)))[0].numpy()
print("corrected original-state mean range:",
      float(mstate125.min()), float(mstate125.max()))

print("\n=== READING-ALIGNED TARGETS ===")
for nm125, ob125 in list(globals().items()):
    if isinstance(ob125, (np.ndarray, torch.Tensor)):
        if ob125.ndim == 1 and len(ob125) == len(mstate125):
            aa125 = ob125.detach().numpy() if isinstance(ob125, torch.Tensor) else ob125
            if np.issubdtype(aa125.dtype, np.number):
                print(nm125, str(aa125.dtype),
                      "RMSE versus corrected means:",
                      float(np.sqrt(np.nanmean((aa125-mstate125)**2))))
    elif hasattr(ob125, "columns") and len(ob125) == len(mstate125):
        print(nm125, "columns:", list(ob125.columns))

print("\n=== ALL-BUNDLE STATE ARRAYS ===")
for nm125, ob125 in list(globals().items()):
    if isinstance(ob125, (np.ndarray, torch.Tensor)):
        if ob125.ndim >= 2 and ob125.shape[0] == len(state_df):
            if tuple(ob125.shape[1:]) in ((992,), (4, 192), (192,)):
                print(nm125, tuple(ob125.shape), str(ob125.dtype))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 125: correct reading query type\n"
            "Found generic decoder type_emb[0] versus trusted type_emb[1]. "
            "Implemented type-one distributional readout and checked identical-input equivalence. "
            "Inspecting retained GPU targets and all-bundle states before fitting. "
            "No weights changed. Python runs: 125; forwards: 15.\n")
```