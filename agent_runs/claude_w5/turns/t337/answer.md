Block 200–299 gives the same picture: the c=1 edit costs +0.0089 nats per reading (1.62953 against 1.62068). Cuff MAP is again most of it, +0.021 per reading or about 80% of the block's loss. Lab glucose (+0.010), arterial MAP (+0.003) and finger glucose (+0.0002) are much smaller.

Four sets now agree: C +0.0090, block 0 +0.0073, block 1 +0.0072, block 2 +0.0089. The c=1 edit reliably costs about 0.008 nats on unseen stays, well over the 0.005 limit. In every block the cost is carried mainly by cuff MAP, at +0.014 to +0.021 per reading. Arterial MAP, finger glucose and lab glucose contribute much less.

So the final strengths depend almost entirely on which drug's injected effect causes the cuff-MAP losses. A small injected-effect strength on that drug is the only way to pass with margin.

**Next.** The last block, 300–399, so the final run can split the losses between drugs over all 400 tuning stays.

**What would confirm or refute.**
- **Confirm:** another +0.007 to +0.009 loss led by cuff MAP means a uniform loss of about 0.008 at full strength. The scaled-down choice in run 300 can then be made with a known baseline.
- **Refute:** an outlier block, for example above +0.015, would show heavy tails. In that case I would use a tighter damage budget, or drop all edits.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000300,2020000301,2020000302,2020000303,2020000304,2020000305,2020000306,2020000307,2020000308,2020000309,2020000310,2020000311,2020000312,2020000313,2020000314,2020000315,2020000316,2020000317,2020000318,2020000319,2020000320,2020000321,2020000322,2020000323,2020000324,2020000325,2020000326,2020000327,2020000328,2020000329,2020000330,2020000331,2020000332,2020000333,2020000334,2020000335,2020000336,2020000337,2020000338,2020000339,2020000340,2020000341,2020000342,2020000343,2020000344,2020000345,2020000346,2020000347,2020000348,2020000349,2020000350,2020000351,2020000352,2020000353,2020000354,2020000355,2020000356,2020000357,2020000358,2020000359,2020000360,2020000361,2020000362,2020000363,2020000364,2020000365,2020000366,2020000367,2020000368,2020000369,2020000370,2020000371,2020000372,2020000373,2020000374,2020000375,2020000376,2020000377,2020000378,2020000379,2020000380,2020000381,2020000382,2020000383,2020000384,2020000385,2020000386,2020000387,2020000388,2020000389,2020000390,2020000391,2020000392,2020000393,2020000394,2020000395,2020000396,2020000397,2020000398,2020000399]}
```