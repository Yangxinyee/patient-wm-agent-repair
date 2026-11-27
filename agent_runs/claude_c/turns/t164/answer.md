With alpha = +300, every readout shrank by about half (−0.92, +0.49, −0.16, −0.23). So the new offset dilutes the dose-dependent part, and this sign brings no offsetting rise in attention.

The sign of the offset's projection on ĥ is unknown, so the token may have been pushed toward "anti-slot" rather than slot-like. I'll run the mirror image: alpha = −302 gives a coefficient −301·(ĥ·b), the mirror of +301·(ĥ·b). If the readouts grow, slot-likeness raises attention and is worth tuning. If they shrink again by a similar amount, the attention share isn't moved by this route and I drop it.

```python
clear_ops(); rank1('encode_action.norm.bias', hhat, -302.0); print(len(ops()))
```