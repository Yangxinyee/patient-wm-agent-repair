**What run 299 showed.**
- The plain c=1 edit is restored exactly (difference 0.0).
- The +0.019 loss of the 1.5×/1.3× version is not spread out. Stays with no drugs are untouched (+0.0007).
- The loss sits in drug stays, almost entirely in readings within one hour of a drug event (+0.038). Readings 1–6 hours later are slightly better than the original (−0.003 to −0.005).
- It is very concentrated: five stays carry 63% of the total.
- So in a few stays, right after drug events, the injected drug effect is far off and confidently wrong. Its intended later effect actually helps.

**What I conclude.** This is a miscalibrated injection at certain drug bundles, not a general leak. The c=1 version injects two-thirds as much, so its cost should be much smaller. Its improvement on training stays (−0.0032) may be consistent with that, but it has never been measured on tuning stays. With one Python run left, I must now establish whether c=1 is safe.

**Next step.** The factual pass on the same tuning stays C, now with the plain c=1 edit. Its hash should be ee4374ba4c, the hash of fwd_6/fwd_8. The original scores 1.65072 here.

**What would confirm or refute.**
- **Confirm:** edited NLL at or below about 1.6527 (within +0.002). I would keep c=1 as the final edit, with readouts as in fwd_6 (insulin x2 ≈ −9.4).
- **Marginal:** between +0.002 and +0.005. I would first check stay set D with another forward pass before deciding.
- **Refute:** above +0.005. The last Python run would then drop all edits, so the submission meets the factual condition.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000111,2020000213,2020000167,2020000348,2020000075,2020000365,2020000114,2020000361,2020000235,2020000017,2020000308,2020000008,2020000094,2020000281,2020000077,2020000188,2020000136,2020000322,2020000399,2020000089,2020000204,2020000245,2020000142,2020000288,2020000190,2020000318,2020000258,2020000378,2020000021,2020000387,2020000143,2020000393,2020000073,2020000007,2020000313,2020000390,2020000398,2020000086,2020000100,2020000222,2020000311,2020000165,2020000260,2020000166,2020000156,2020000218,2020000262,2020000140,2020000287,2020000329,2020000307,2020000051,2020000367,2020000193,2020000334,2020000374,2020000341,2020000280,2020000261,2020000340,2020000328,2020000392,2020000093,2020000300,2020000212,2020000296,2020000118,2020000326,2020000053,2020000148,2020000199,2020000183,2020000000,2020000106,2020000145,2020000015,2020000237,2020000068,2020000252,2020000209,2020000101,2020000202,2020000041,2020000397,2020000269,2020000091,2020000243,2020000058,2020000317,2020000046,2020000279,2020000248,2020000079,2020000303,2020000251,2020000371,2020000187,2020000272,2020000002,2020000192]}
```