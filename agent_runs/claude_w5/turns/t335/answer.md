Block 0–99 confirms the c=1 edit loses factual fit outside set C too: 1.64661 against 1.63933 for the original, +0.0073 nats. Almost all of it is cuff MAP, which got +0.0177 worse and makes up about 89% of the total. Finger glucose (+0.0012), arterial MAP (+0.0017) and lab glucose (0.0000) barely move. In set C, glucose losses were more visible but cuff MAP was still the largest share, 62%.

**What I conclude.**
- The damage is general across stays, not a quirk of set C.
- Its main carrier is cuff MAP. So whether insulin can keep a usefully high strength depends on whether the cuff losses follow norepinephrine or insulin.
- If they follow norepinephrine, the final run can cut norepinephrine strongly and keep insulin, which drives the main readout, fairly high. If they follow insulin, insulin must be cut too.

**Next.** I continue the c=1 factual pass over the remaining tuning blocks; this one is ids 100–199. Run 300 can then split the loss between the two drugs and pick strengths on all 400 tuning stays rather than 100 or 200.

**What would confirm or refute.** If block 100–199 also shows a loss of about +0.007 to +0.009, dominated by cuff MAP, the pattern is stable and the split between drugs can be trusted. A very different block would mean rare outlier stays drive the loss. That would call for a smaller damage budget in the final choice.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000100,2020000101,2020000102,2020000103,2020000104,2020000105,2020000106,2020000107,2020000108,2020000109,2020000110,2020000111,2020000112,2020000113,2020000114,2020000115,2020000116,2020000117,2020000118,2020000119,2020000120,2020000121,2020000122,2020000123,2020000124,2020000125,2020000126,2020000127,2020000128,2020000129,2020000130,2020000131,2020000132,2020000133,2020000134,2020000135,2020000136,2020000137,2020000138,2020000139,2020000140,2020000141,2020000142,2020000143,2020000144,2020000145,2020000146,2020000147,2020000148,2020000149,2020000150,2020000151,2020000152,2020000153,2020000154,2020000155,2020000156,2020000157,2020000158,2020000159,2020000160,2020000161,2020000162,2020000163,2020000164,2020000165,2020000166,2020000167,2020000168,2020000169,2020000170,2020000171,2020000172,2020000173,2020000174,2020000175,2020000176,2020000177,2020000178,2020000179,2020000180,2020000181,2020000182,2020000183,2020000184,2020000185,2020000186,2020000187,2020000188,2020000189,2020000190,2020000191,2020000192,2020000193,2020000194,2020000195,2020000196,2020000197,2020000198,2020000199]}
```