The descriptive fits suggest that persistence is already substantial; the main problem is weak immediate dose sensitivity. I’ll test whether prelude attention is suppressing the action token by temporarily making that attention uniform. Stronger responses in the record-supported directions would justify testing a partial, factual-safe version.

```python
from pathlib import Path
import json

while len(ops()) > 1:
    undo()

scale("core.prelude.0.attn.q.weight", 0.0)
print("DIAGNOSTIC CANDIDATE:", ops())
print("Probe stays:", len(probe_ids), probe_ids[:3], probe_ids[-3:])

Path("/work/uniform_prelude_ops.json").write_text(json.dumps(ops(), indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 93: uniform-prelude attention diagnostic\n"
            "- Rollout impulse-response fits indicate weak immediate drug response, not inadequate persistence.\n"
            "- Numeric gain 4 retained; set core.prelude.0.attn.q.weight to zero, making prelude attention uniform.\n"
            "- Diagnostic only: factual safety is unknown. Testing whether learned attention excludes action information.\n"
            "- A materially stronger negative insulin and positive NE response would justify a partial, factual-tested edit.\n"
            "- Forward passes used: 34; 16 remain.\n")
```