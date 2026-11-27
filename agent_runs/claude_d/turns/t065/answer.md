The first half confirms the pipeline. Doses enter only through the action atoms' embedding (fine rate/dose bins plus the z_dose/z_rate numeric columns) and `encode_action`, which gives u_k. u_k then feeds z_post and the prelude token. So an edit on the action pathway changes nothing on bundles without drugs, and acts on later bundles only through the state. Next I'll read the heads and dose-edit sections, to see exactly how predicted means and the rollout are formed.

```python
print(arch[i:])
```