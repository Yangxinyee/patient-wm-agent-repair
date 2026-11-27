With the NE kernel aligned correctly, the NE targets reproduce the notebook (sel1: ne_x2 +4.87, ne_half −2.44). Both insulin kernel files are identical. They give sel1 insulin targets of −14.35 / +7.17, about 2.5% larger than the notebook's −13.99 / +6.99, which were probably computed before the minute-based time-alignment fix. That is within the uncertainty of the effect estimate itself.

Pooled over sel1 and sel2 (row-weighted), current edit against record-based targets:

| Readout | Model | Target | Error |
|---|---|---|---|
| insulin x2 (main) | −13.70 | −13.87 | about 1% low |
| insulin x1/2 | +6.69 | +6.93 | about 3.5% low |
| NE x1/2 | −2.21 | −2.39 | about 7.5% low |
| NE x2 | +1.47 | +4.77 | structural limit above the cap |

Both insulin errors are within 1 standard error on each set. So the insulin calibration generalises to unseen stays.

The remaining open question is the factual-fit constraint, which decides whether the edit counts at all. It also decides whether there is room to raise the NE head-channel gain from 2 to about 2.2, which would close most of the NE x1/2 gap. I'll measure value NLL, edited vs original, on all 400 tuning stays in four passes, starting with sel1.

The constraint holds if the edited-minus-original NLL is comfortably below +0.005. Earlier offline estimates suggest about +0.002.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000, 2020000007, 2020000008, 2020000013, 2020000023, 2020000024, 2020000025, 2020000032, 2020000034, 2020000035, 2020000039, 2020000040, 2020000041, 2020000053, 2020000060, 2020000064, 2020000068, 2020000069, 2020000076, 2020000079, 2020000081, 2020000084, 2020000091, 2020000094, 2020000097, 2020000102, 2020000105, 2020000112, 2020000117, 2020000120, 2020000124, 2020000126, 2020000128, 2020000132, 2020000134, 2020000142, 2020000147, 2020000148, 2020000149, 2020000154, 2020000156, 2020000158, 2020000171, 2020000174, 2020000176, 2020000181, 2020000182, 2020000183, 2020000187, 2020000188, 2020000194, 2020000198, 2020000204, 2020000211, 2020000212, 2020000214, 2020000221, 2020000229, 2020000232, 2020000234, 2020000248, 2020000249, 2020000252, 2020000262, 2020000264, 2020000267, 2020000268, 2020000276, 2020000279, 2020000283, 2020000286, 2020000293, 2020000298, 2020000300, 2020000304, 2020000305, 2020000308, 2020000309, 2020000315, 2020000327, 2020000341, 2020000342, 2020000348, 2020000351, 2020000354, 2020000357, 2020000360, 2020000361, 2020000363, 2020000368, 2020000369, 2020000370, 2020000378, 2020000381, 2020000384, 2020000386, 2020000388, 2020000396, 2020000398, 2020000399]}
```