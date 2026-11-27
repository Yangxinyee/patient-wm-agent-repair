Run 294 settled three things:

* **Which edit is in place.** The kv delta is not 0.5× the earlier full edit, so the rebuilt C1 version, with very large weights, is what is installed.
* **No factual check yet.** fwd_7 was a states pass, so this edit's factual fit has never been checked.
* **Hourly dose_edit profile:**
  * For insulin, the edit adds only about −1 to the doubling shift. It slightly weakens the halving shift.
  * For norepinephrine, both shifts are exactly 0.0 at every hour, where the original model's shift grows to about 0.6. That looks like the edited MAP prediction is pinned (saturated), so it no longer responds to the dose.

My working conclusion is that the NE channel of this edit is broken and should be switched off, and the insulin channel is too weak. With only 6 Python runs left, I'll use a forward pass first, since it costs no Python run. This is a factual "both" pass on the 100 tuning stays, to see the per-event NLL change of the current edit.

* **Confirmed** if map_arterial NLL gets clearly worse while glucose NLL is about neutral or better. Then the NE channel is the problem, and the next build drops it and raises the insulin gain.
* **Refuted** if glucose NLL also gets worse. Then the insulin channel leaks too, and I should scale it down or revert.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000, 2020000008, 2020000013, 2020000018, 2020000019, 2020000023, 2020000032, 2020000038, 2020000039, 2020000040, 2020000041, 2020000048, 2020000052, 2020000060, 2020000063, 2020000069, 2020000071, 2020000074, 2020000076, 2020000081, 2020000084, 2020000086, 2020000089, 2020000091, 2020000094, 2020000102, 2020000111, 2020000112, 2020000117, 2020000120, 2020000124, 2020000126, 2020000130, 2020000132, 2020000154, 2020000156, 2020000158, 2020000165, 2020000171, 2020000176, 2020000181, 2020000182, 2020000183, 2020000187, 2020000188, 2020000198, 2020000201, 2020000204, 2020000211, 2020000212, 2020000214, 2020000232, 2020000249, 2020000250, 2020000252, 2020000254, 2020000264, 2020000265, 2020000267, 2020000268, 2020000001, 2020000003, 2020000010, 2020000016, 2020000022, 2020000024, 2020000025, 2020000026, 2020000027, 2020000033, 2020000035, 2020000042, 2020000045, 2020000046, 2020000047, 2020000049, 2020000051, 2020000054, 2020000057, 2020000059, 2020000007, 2020000012, 2020000015, 2020000031, 2020000034, 2020000043, 2020000050, 2020000053, 2020000056, 2020000058, 2020000066, 2020000067, 2020000079, 2020000093, 2020000095, 2020000096, 2020000097, 2020000110, 2020000125, 2020000127]}
```