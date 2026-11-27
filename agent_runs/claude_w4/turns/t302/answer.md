Three findings, two of which rule out my leading suspects:

- **Centring was right when v4 was built.** v4 used 1.77 as the insulin centring rate, as intended. Only the live global `rbar` (0.52) is wrong, and I will reset it before any rebuild. The channel directions are the stored ones, and `c` is restored to 6.
- **The written vectors are not oversized.** Per unit of insulin signal, `b_in` writes a vector of norm 1.1, and 4.1 for NE. Normal bundle-to-bundle variation of `b_in` output is 2.9. So "huge writes in invisible directions" is not the explanation.
- **The reading-bundle leak is real but small by design.** It is 0.13 units of insulin signal per bundle, which by the design should give about −0.11 mg/dL per bundle. That is small and has the wrong sign to explain a +2 mg/dL-per-hour upward drift.

So my CPU accounting cannot explain what the rollout shows. The realised effect of the edit differs from the designed one, and I need to measure it directly.

The most useful single measurement is a teacher-forced factual run comparing both models, on the sel100 stays (the list printed in run 265). It gives:
1. the factual NLL cost of v4, which is the hard constraint;
2. the exact per-reading change in predicted glucose and MAP.

From those changes I can regress the change at each reading on the change at the previous reading and on the doses in between. That separates three pushes: from reading bundles alone (the intercept), per unit of centred insulin, and per unit of centred NE.

How to read the outcome:
- A clearly positive intercept means a dose-independent leak.
- A positive insulin coefficient means the dose channel acts with reversed sign.
- A negative insulin coefficient near −0.5 per unit means the channel works as designed in teacher-forced mode. In that case the problem arises only in the free-running rollout.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000001, 2020000002, 2020000005, 2020000006, 2020000009, 2020000012, 2020000014, 2020000024, 2020000025, 2020000026, 2020000029, 2020000035, 2020000042, 2020000043, 2020000049, 2020000051, 2020000057, 2020000064, 2020000066, 2020000073, 2020000075, 2020000076, 2020000079, 2020000085, 2020000088, 2020000089, 2020000090, 2020000091, 2020000097, 2020000099, 2020000101, 2020000103, 2020000109, 2020000111, 2020000112, 2020000114, 2020000117, 2020000119, 2020000120, 2020000126, 2020000131, 2020000133, 2020000135, 2020000138, 2020000141, 2020000144, 2020000145, 2020000154, 2020000158, 2020000159, 2020000164, 2020000169, 2020000170, 2020000171, 2020000174, 2020000177, 2020000182, 2020000188, 2020000197, 2020000200, 2020000215, 2020000220, 2020000223, 2020000224, 2020000227, 2020000231, 2020000233, 2020000236, 2020000238, 2020000240, 2020000247, 2020000249, 2020000253, 2020000257, 2020000259, 2020000262, 2020000274, 2020000277, 2020000280, 2020000300, 2020000301, 2020000304, 2020000310, 2020000311, 2020000317, 2020000319, 2020000321, 2020000322, 2020000325, 2020000328, 2020000334, 2020000335, 2020000355, 2020000361, 2020000362, 2020000365, 2020000369, 2020000377, 2020000386, 2020000387]}
```