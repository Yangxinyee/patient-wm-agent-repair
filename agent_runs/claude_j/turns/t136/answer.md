Two findings. First, a bug: the tool's factual table is ordered differently from my reading table (events match on only 21% of rows). So the residual slopes printed (0.92, −2.03, −0.96) are meaningless; I'll merge on (stay, hour, event) from now on.

Second, and more important: a linear readout of the head query q recovers the total insulin history RI well (CV R² 0.89) but the exogenous part TI barely at all (CV R² 0.15). The model's state hardly encodes the random part of the dose. A correction read from q cannot tell a doubled dose from a high glucose. I need an explicit dose channel through the recurrent state. To design one, I need the exact state update (flow decay, the `h` update, and the coda that rewrites `h`). I'll grep only those lines of ARCHITECTURE.md.

```python
arch_=open('/model/ARCHITECTURE.md').read().splitlines()
keys_=['coda','c_out','a_bar','delta_in','b_in','a_in','flow','slots','out_norm','memory','mean_slots','h =','h=']
hit=[i for i,l in enumerate(arch_) if any(k in l for k in keys_)]
print(len(arch_), 'lines;', len(hit), 'hits')
for i in hit[:30]: print(i, arch_[i][:160])
```