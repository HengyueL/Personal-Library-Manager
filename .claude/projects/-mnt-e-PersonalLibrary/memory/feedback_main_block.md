---
name: Keep __main__ block minimal
description: User wants __main__ blocks to only call main(), not contain logic
type: feedback
---

Keep `if __name__ == "__main__":` blocks to a single `main()` call. Put all logic (argparse, business logic, output) inside a `def main()` function. Use `logger` (not `print`) for all output in scripts.

**Why:** User preference — they've corrected both of these.

**How to apply:** Whenever writing a script, define `def main()`, keep `__main__` to just `main()`, and use `logging`/`logger` for all output instead of `print`.
