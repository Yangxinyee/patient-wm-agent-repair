Run 300, my last Python run, saved the final edit.

**What it showed.**
- The scaling family rebuilt exactly. The insulin and norepinephrine directions are clean, symmetric projections, applied to units 131, 140 and 502.
- Splitting the c=1 loss over all 400 tuning stays settled the key question. About 90% of it (125 of 140 nats) sits in readings within an hour of a norepinephrine event with no insulin event. That includes almost all the cuff-MAP loss (95 nats).
- Readings right after insulin-only events carry only about 9 nats. Readings after both drugs show a small benefit.
- On set C, both injected effects grew faster than quadratically with strength, so no fixed cost was detected.

**What I conclude.** The norepinephrine part of my edit is what breaks the factual fit; the insulin part is nearly harmless. So the final run kept insulin at full c=1 strength and cut norepinephrine to 0.3. The predicted factual cost is +0.0019 nats on the 400 tuning stays, with deliberately conservative extrapolation. The expected readouts are:

| Readout | Expected | Target |
|---|---|---|
| Insulin x2 | ≈ −9.4 | −14.3 |
| Insulin x1/2 | ≈ −1.6 | +7.15 |
| Norepinephrine x2 | ≈ +1.1 | +4.11 |
| Norepinephrine x1/2 | ≈ −0.24 | −2.05 |

No Python runs remain, so this is the submitted state.

**Next.** I'll measure its factual cost on all 400 tuning stays, block by block, starting with ids 0–99, which cost +0.0073 at full strength. Then a dose_edit pass will give the expected readouts.

**What would confirm or refute.** The scaling model predicts roughly +0.001 to +0.002 per block; the realistic faster-than-quadratic shrinkage suggests even less. A value well under 0.005 confirms that the norepinephrine part caused the damage and that the edit is valid. A value near or above 0.005 would refute it, although the edit can no longer be changed.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000,2020000001,2020000002,2020000003,2020000004,2020000005,2020000006,2020000007,2020000008,2020000009,2020000010,2020000011,2020000012,2020000013,2020000014,2020000015,2020000016,2020000017,2020000018,2020000019,2020000020,2020000021,2020000022,2020000023,2020000024,2020000025,2020000026,2020000027,2020000028,2020000029,2020000030,2020000031,2020000032,2020000033,2020000034,2020000035,2020000036,2020000037,2020000038,2020000039,2020000040,2020000041,2020000042,2020000043,2020000044,2020000045,2020000046,2020000047,2020000048,2020000049,2020000050,2020000051,2020000052,2020000053,2020000054,2020000055,2020000056,2020000057,2020000058,2020000059,2020000060,2020000061,2020000062,2020000063,2020000064,2020000065,2020000066,2020000067,2020000068,2020000069,2020000070,2020000071,2020000072,2020000073,2020000074,2020000075,2020000076,2020000077,2020000078,2020000079,2020000080,2020000081,2020000082,2020000083,2020000084,2020000085,2020000086,2020000087,2020000088,2020000089,2020000090,2020000091,2020000092,2020000093,2020000094,2020000095,2020000096,2020000097,2020000098,2020000099]}
```