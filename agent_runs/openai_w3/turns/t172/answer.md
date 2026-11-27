The explicit signals are behaving as intended: posterior compensation is accurate on held-out stays, the carrier can be read with little error, and its attention gate remains close to one. Those are local checks, not proof of a successful repair. Next I’ll verify the recurrent replay and carrier-only disturbance, then calibrate where the signal enters the state.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def prelude155(tokens, overrides=None, injection=None):
    overrides = {} if overrides is None else overrides
    def w155(name):
        return tensor153(overrides[name] if name in overrides else load_original(name))
    x155 = tensor153(tokens)
    p155 = "core.prelude.0"
    n155 = F.layer_norm(x155, (192,), w155(p155+".norm1.weight"),
                        w155(p155+".norm1.bias"))
    q155 = F.linear(n155[:,:4], w155(p155+".attn.q.weight")).reshape(
        -1,4,4,48).transpose(1,2)
    k155,v155 = F.linear(n155, w155(p155+".attn.kv.weight")).chunk(2,-1)
    k155 = k155.reshape(-1,7,4,48).transpose(1,2)
    v155 = v155.reshape(-1,7,4,48).transpose(1,2)
    att155 = torch.softmax(q155 @ k155.transpose(-1,-2)/np.sqrt(48), -1)
    pool155 = (att155 @ v155).transpose(1,2).reshape(-1,4,192)
    after155 = x155[:,:4] + F.linear(pool155, w155(p155+".attn.out.weight"))
    if injection is not None:
        after155 = after155 + tensor153(injection)
    norm155 = F.layer_norm(after155, (192,), w155(p155+".norm3.weight"),
                          w155(p155+".norm3.bias"))
    hidden155 = F.gelu(F.linear(norm155, w155(p155+".ffn.0.weight"),
                              w155(p155+".ffn.0.bias")))
    raw155 = after155 + F.linear(hidden155, w155(p155+".ffn.2.weight"),
                               w155(p155+".ffn.2.bias"))
    E155 = F.layer_norm(raw155, (192,), w155("core.prelude_norm.weight"),
                       w155("core.prelude_norm.bias"))
    return E155, after155, n155, att155

slots155 = tensor153(load_original("core.slots"))
tokens_original155 = torch.cat((
    tensor153(h81)+slots155, tensor153(e81)[:,None],
    tensor153(u81)[:,None],
    F.linear(tensor153(Ptarget154[:,:32]),
             tensor153(zin_original154), tensor153(zb_original154))[:,None]), 1)
tokens_carrier155 = torch.cat((
    tensor153(h81)+slots155, tensor153(e81)[:,None],
    tensor153(u81)[:,None],
    tensor153(raw_latent154)[:,None]), 1)

E_original155,_,_,_ = prelude155(tokens_original155)
E_zero155,_,_,_ = prelude155(tokens_original155, zero_channel147)
E_carrier155, after_carrier155, norm_carrier155, attention_carrier155 = prelude155(
    tokens_carrier155, zero_channel147)

print("Original prelude replay RMSE:",
      float(torch.sqrt(torch.mean((E_original155-tensor153(E81))**2))))
print("Zero-head original-input E RMSE:",
      float(torch.sqrt(torch.mean((E_zero155-E_original155)**2))))
print("Carrier-only E RMSE:",
      float(torch.sqrt(torch.mean((E_carrier155-E_original155)**2))))
select155 = attention_carrier155[:,3,:,6].numpy()
print("Reserved head latent attention min/mean/max:",
      float(select155.min()), float(select155.mean()), float(select155.max()))
print("Other-head mean latent attention:",
      attention_carrier155[:,:3,:,6].mean((0,2)).numpy())

print("\nState-file keys:", list(S.keys()))
print("Existing calibration positions:",
      "pick105", np.asarray(pick105).tolist(),
      "abs105", np.asarray(abs105).tolist())
positions155 = np.asarray(pick105, dtype=int)
indices155 = np.asarray(idx81)[positions155]
print("idx81[pick105] matches abs105:", np.array_equal(indices155, np.asarray(abs105)))
print("memory helper constants:", repr(memory82.__code__.co_consts)[:500])
print("attention helper constants:", repr(attention82.__code__.co_consts)[:700])

state_checks155 = []
for include155 in (False, True):
    mem155, mask155, age155 = memory82(indices155, include155)
    hs155, os155 = state_update82(
        tensor153(h81)[positions155], E_original155[positions155],
        mem155, mask155, age155)
    observed_h155 = tensor153(S["h"])[indices155].reshape(-1,4,192)
    observed_o155 = tensor153(S["out"])[indices155]
    eh155 = float(torch.sqrt(torch.mean((hs155-observed_h155)**2)))
    eo155 = float(torch.sqrt(torch.mean((os155-observed_o155)**2)))
    state_checks155.append((eo155, include155, mem155, mask155, age155))
    print("Memory include-current", include155,
          "h RMSE", eh155, "out RMSE", eo155,
          "age range", float(age155.min()), float(age155.max()))
best_memory155 = min(state_checks155, key=lambda x:x[0])
memory155, mask155, age155 = best_memory155[2:]

print("\nRetained readout/calibration scale:")
print("directions136 norms", np.linalg.norm(np.asarray(directions136),axis=0))
print("G136 norms", np.linalg.norm(np.asarray(G136),axis=1))
print("T136", np.asarray(T136))
print("read_codes105", np.asarray(read_codes105))
print("next_dt105", getattr(next_dt105,"shape",None))
print("next_view105 constants", repr(next_view105.__code__.co_consts)[:600])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 155; forwards 10. Checked original and "
        "carrier-only prelude replay, reserved-head selection, and "
        "recurrent replay against authoritative states. No candidate "
        "posterior/carrier or dose-value projection installed yet. "
        "Current factual-neutral zero-payload head edit remains.\n"
    )
```