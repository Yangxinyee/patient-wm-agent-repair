Run 297 made no edit, and the five edited tensors are unchanged.

**Targets.** My notes confirm the targets I am aiming at:

| Readout | Target |
|---|---|
| Insulin x2 (glucose) | −14.3 |
| Insulin x1/2 (glucose) | +7.15 |
| Norepinephrine x2 (MAP) | +4.11 |
| Norepinephrine x1/2 (MAP) | −2.05 |

**b_in difference.** The current b_in differs from the c=1 version (kept in the kernel as B_IN_c1) in every row and column. So the gain scaling is a dense change, not a scaling of a few columns. The direction vectors q_I and q_N are still in the kernel, so I could retune the gains later if needed.

**New test stays.** Tuning stay ids run from 2020000000 to 2020000399. The script could not recognise insulin or norepinephrine events in the tuning file, so it picked sets C and D at random. Each holds 100 stays not in selA. That is harmless; random stays are close to how the evaluation draws them.

**What I conclude.** The saved edit is the c_I=1.5, c_N=1.3 version. It has never had a factual check: only the c=1 version did, scoring 0.0032 nats better than the original. Raising the gains could cost some factual fit, and that is a hard constraint.

**Next step.** A teacher-forced factual pass on set C, original and edited models together. The reported model hash should also be a56a7f6079, which confirms nothing changed since fwd_9.

**What would confirm or refute.**
- **Confirm:** the edited model's NLL per reading is at most about +0.002 worse than the original's, ideally better. The current version is then safe, and I go on to replicate the dose readouts on C.
- **Refute:** a cost above about +0.003 would leave too little margin. I would then fall back toward c=1 using B_IN_c1. That trades accuracy on the main readout for safety.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000111,2020000213,2020000167,2020000348,2020000075,2020000365,2020000114,2020000361,2020000235,2020000017,2020000308,2020000008,2020000094,2020000281,2020000077,2020000188,2020000136,2020000322,2020000399,2020000089,2020000204,2020000245,2020000142,2020000288,2020000190,2020000318,2020000258,2020000378,2020000021,2020000387,2020000143,2020000393,2020000073,2020000007,2020000313,2020000390,2020000398,2020000086,2020000100,2020000222,2020000311,2020000165,2020000260,2020000166,2020000156,2020000218,2020000262,2020000140,2020000287,2020000329,2020000307,2020000051,2020000367,2020000193,2020000334,2020000374,2020000341,2020000280,2020000261,2020000340,2020000328,2020000392,2020000093,2020000300,2020000212,2020000296,2020000118,2020000326,2020000053,2020000148,2020000199,2020000183,2020000000,2020000106,2020000145,2020000015,2020000237,2020000068,2020000252,2020000209,2020000101,2020000202,2020000041,2020000397,2020000269,2020000091,2020000243,2020000058,2020000317,2020000046,2020000279,2020000248,2020000079,2020000303,2020000251,2020000371,2020000187,2020000272,2020000002,2020000192]}
```