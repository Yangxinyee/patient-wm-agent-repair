Scaling head 3 ×4 made insulin ×2 a little stronger (−2.05) but the half-dose weaker (+0.60), and pushed both NE readouts more negative (−0.98 / −1.11). The per-head contributions don't add up linearly. Scaling any one head moves the readouts by only ±0.1–0.3, most likely because scaling a head also shifts the attention output's baseline. Head 0 is still the only one that clearly strengthens the insulin response.

I can't pursue head 0 without knowing whether it also carries reading information, which would cost factual NLL. So next I'll check the factual NLL of the head-0 ×4 model. A rise well under 0.005 makes head 0 a usable lever to combine with the input-side scaling. A large rise means it carries readings too and isn't selective.

```python
clear_ops(); scale_head(0, 4.0); print(len(ops()))
with open('/work/notes.md','a') as f:
    f.write("- Passes 18-21 head scaling x4 (prelude attn.out cols): h0 -2.53/+1.67/-0.36/-0.97; h1 -1.50/+0.45/-0.37/-0.52; h2 -1.48/+0.54/-0.77/-0.71; h3 -2.05/+0.60/-0.98/-1.11. Nonlinear (base shifts). h0 best.\n")
```