The `core` tensor list shows the E→h path is all shared linear or per-dimension maps (b_in, delta_in, a_in, flow.a, prelude_norm). As the earlier step-variance check showed, reading bundles dominate every h direction, so none of these can amplify only the action-driven part.

The only levers that move the readouts at an affordable cost are the numeric dose/rate columns and head0 of the prelude attention. So I will look for the best mix of these two under the 0.005 limit. Numeric x5 cost 0.0016; adding head0 x2.5 brought the total to 0.0051, too much.

Next I will build numeric x5 + head0 x2.0 and measure its readouts. I expect insulin x2 around -2.8 and insulin half around +2.1, at a projected factual cost of 0.003-0.004. A follow-up factual pass will confirm the cost.

```python
clear_ops()
e6=np.eye(6)
for j in [2,4]: rank1('embed.numeric.weight',e6[j],4.0,side='in')
eh=np.eye(192)
for j in range(48): rank1('core.prelude.0.attn.out.weight',eh[j],1.0,side='in')
print(len(ops()))
```