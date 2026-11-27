The age embedding core.age is a small MLP on log(1+age), added to the cross-attention keys over the memory M. A full CPU replica of the step therefore needs every bundle's e vector, including measurement bundles, where my replica was about 33% off.

Before spending runs on a full replica, I'll check whether the fwd_4 probe gives GPU ground truth for some e_k coordinates. In that probe, z_mean copied 32 input coordinates at every bundle. If those coordinates fall inside e_k, I can use them to debug the measurement-atom embedding.

```python
Z4r = torch.tensor(Z4[rows]).float()
best = []
for o in range(0, 576-31):
    err = (Z4r - XIN3[:, o:o+32].float()).abs().mean().item()
    best.append((err, o))
best.sort(); print(best[:3])
```