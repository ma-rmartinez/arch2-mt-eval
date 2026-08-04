# submission/

Put your candidate eval set here as **`submission/eval_set.json`**.

The scorer reads exactly that path by default (override with `ARCH_SUBMISSION`).
The schema is documented in `.arch/worker_README.md` under "Submission contract"
and enforced by `eval/schema.py` — validation is strict, and an item that cannot
be scored unambiguously is rejected rather than silently counted as a miss.

To start from the worked example:

    python3 scripts/build_starter_eval_set.py -o submission/eval_set.json

Then extend it. The starter set has 7 identifying items against the 120 needed
for full size credit, so it scores badly on purpose.
