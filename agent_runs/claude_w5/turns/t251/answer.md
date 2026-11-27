The local and tool edited models disagree. In my float64 sequential local run, the 13:00 offset averages +0.12 and correlates −0.45 with the insulin rate, the intended direction. The tool's offset averages +2.37 and correlates +0.43. The two correlate only 0.30 with each other. So either the tool computes my edited weights differently from my local implementation, or its rollout hides more than I assume. The local implementation reproduced the tool's original-model states to about 1e-6, so whatever differs must only matter once the edited units are active.

To find where they diverge, I'll ask for the tool's states from both models on the same tuning stays. Then I can compare, bundle by bundle, with my local original and edited trajectories.
- If the tool's edited states match local to about 1e-5, the computation is the same. The problem would then be the rollout protocol, for example readings hidden earlier than I think.
- If they diverge, the first divergent bundle and the direction of the difference will show which part of my edit the tool evaluates differently: gate units, carrier direction, or b_in gain.

```forward
{"type": "states", "model": "both", "split": "tuning", "stays": [2020000014, 2020000033, 2020000034, 2020000036, 2020000038, 2020000045, 2020000054, 2020000066, 2020000070, 2020000090, 2020000092, 2020000112, 2020000117]}
```