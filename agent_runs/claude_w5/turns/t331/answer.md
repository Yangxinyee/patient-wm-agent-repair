Run 298 rewrote the edit in a cleaner form:

- **b_in back to the c=1 version.** That version is itself edited, up to 4.69 away from the original.
- **Gain increase recovered exactly.** The b_in rescaling decomposes as a rank-2 matrix with singular values 0.5 and 0.3, which are exactly c_I−1 and c_N−1. The residual is 2e-6.
- **Gain applied only at the source.** Only three units of prelude.0's feed-forward layer carry my injection, and all three were repurposed for it. Their output columns got the (I+M) gain, a 41% change in those columns. Nothing else in the model's state is now amplified.
- **No layer in between.** There is only one prelude layer, and no norm sits between it and b_in, so this should act like the b_in rescaling on the drug effect alone.

**What I conclude.** If the +0.019 factual cost came from collateral amplification by b_in, it should now mostly disappear. Only the deliberate 1.5×/1.3× stronger drug effect would remain, which I expect to cost at most a few thousandths of a nat.

**Next step.** The same factual pass on tuning stays C, original and edited models together. The original scored 1.65072 there.

**What would confirm or refute.**
- **Confirm:** the edited NLL is at or below about 1.6527 (within +0.002), ideally below 1.65072. The cuff MAP and lab glucose losses should be mostly gone. I would then run dose_edit on C to check insulin x2 is still near −14.
- **Refute:** still above about 1.6557 (+0.005). Then the stronger effect itself, or the c=1 b_in edit, is harmful on these stays. I would fall back to the plain c=1 version from the backup, and check it the same way.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000111,2020000213,2020000167,2020000348,2020000075,2020000365,2020000114,2020000361,2020000235,2020000017,2020000308,2020000008,2020000094,2020000281,2020000077,2020000188,2020000136,2020000322,2020000399,2020000089,2020000204,2020000245,2020000142,2020000288,2020000190,2020000318,2020000258,2020000378,2020000021,2020000387,2020000143,2020000393,2020000073,2020000007,2020000313,2020000390,2020000398,2020000086,2020000100,2020000222,2020000311,2020000165,2020000260,2020000166,2020000156,2020000218,2020000262,2020000140,2020000287,2020000329,2020000307,2020000051,2020000367,2020000193,2020000334,2020000374,2020000341,2020000280,2020000261,2020000340,2020000328,2020000392,2020000093,2020000300,2020000212,2020000296,2020000118,2020000326,2020000053,2020000148,2020000199,2020000183,2020000000,2020000106,2020000145,2020000015,2020000237,2020000068,2020000252,2020000209,2020000101,2020000202,2020000041,2020000397,2020000269,2020000091,2020000243,2020000058,2020000317,2020000046,2020000279,2020000248,2020000079,2020000303,2020000251,2020000371,2020000187,2020000272,2020000002,2020000192]}
```