The info dump gave me what I needed:
- The tuning stays are ids 2020000000 to 2020000399, and the 100 selected stays (sel100) are a subset of them, so requests use split "tuning".
- My notebook's record-based truth on sel100 is insulin x2 −12.96 and x1/2 +6.48, norepinephrine x2 +4.94 and x1/2 −2.47. The original model gives only −0.35, +0.21, +0.02 and −0.25, so it barely responds to dose.
- The full-strength gains calibrated by an earlier dose_edit were kI≈4.85 and kN≈6.9. v10b uses 60% of that for insulin and 50% for norepinephrine.
- A backup of v10b is saved in /work.

I expect v10b to give roughly −7.9 for insulin x2 and +2.5 for norepinephrine x2. Its factual cost after the first drug should be about zero, according to my earlier cost curve (zero near 60% strength).

The open question is whether removing the readout leak at the first bundle got rid of the large cost before the first drug, which was +0.17 to +0.25 nats in every earlier version. I'll test that first with a paired factual pass on the same 100 stays as before, so the before/after comparison is direct.
- **Confirmed:** total value-NLL change within about ±0.003, with the per-event "pre" parts near zero.
- **Refuted:** a change of +0.05 or more again. That would mean the leak is still there, and I would reconsider the edit before spending a Python run.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000001,2020000002,2020000005,2020000006,2020000009,2020000012,2020000014,2020000024,2020000025,2020000026,2020000029,2020000035,2020000042,2020000043,2020000049,2020000051,2020000057,2020000064,2020000066,2020000073,2020000075,2020000076,2020000079,2020000085,2020000088,2020000089,2020000090,2020000091,2020000097,2020000099,2020000101,2020000103,2020000109,2020000111,2020000112,2020000114,2020000117,2020000119,2020000120,2020000126,2020000131,2020000133,2020000135,2020000138,2020000141,2020000144,2020000145,2020000154,2020000158,2020000159,2020000164,2020000169,2020000170,2020000171,2020000174,2020000177,2020000182,2020000188,2020000197,2020000200,2020000215,2020000220,2020000223,2020000224,2020000227,2020000231,2020000233,2020000236,2020000238,2020000240,2020000247,2020000249,2020000253,2020000257,2020000259,2020000262,2020000274,2020000277,2020000280,2020000300,2020000301,2020000304,2020000310,2020000311,2020000317,2020000319,2020000321,2020000322,2020000325,2020000328,2020000334,2020000335,2020000355,2020000361,2020000362,2020000365,2020000369,2020000377,2020000386,2020000387]}
```