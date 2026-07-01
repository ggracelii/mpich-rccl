# Musashi's Way

Forge code like Musashi's blade—one swing, perfect. Adapt, strike, merge. Empty mind, full trunk. No copies, no noise. Sharpen. Always sharpen. Ten lines beat a hundred if they kill doubt. Be the ring, not the ripple.

Edge cases don't get into master—ever.
Code repeats? Fix the root, not the symptom.
Silence means you swallowed an error. Throw it.
If you add lines to refactor, you didn't learn.
PR waits >24h? You're buried.
Ask why? You never read.
Master dies when entropy slips. Fight. Always.

> **About**: Actionable principles only. No tutorials, no implementation details.
> **Update**: PR when patterns proven (2+ uses), standards evolve, or conflicts arise.

## Discipline Defaults

- **Succinct** - Every line must drive action; delete repetition and narrative.
- **Actionable** - State rules, patterns, expectations. Implementation detail lives elsewhere.
- **Current** - When tooling shifts, remove stale guidance in the same PR.
- **High-level** - Describe what to do; link to where the how lives.

**Include**: Principles, guardrails, cross-links to deep docs.
**Exclude**: Tutorials, historical debates, TODOs.

## Philosophy: Five Rings

- **Earth (Foundation)** - Types, naming, structure: strict enforcement, zero exceptions
- **Water (Adaptability)** - Codify vs AGENTS.md: enforce what breaks code, guide what improves it
- **Fire (Decisive Action)** - Seize the initiative: throw on error, surface immediately, fail fast
- **Wind (Observation)** - Study other approaches, understand tradeoffs, critique and evolve
- **Void (Emptiness)** - No dead code, no clever tricks, drop ego: what is needed to be

---

## Core Rules (Earth — Non-Negotiable)

### Forbidden Patterns

- Silently eating errors — fail fast, never swallow exceptions
- `lint-disable` comments — fix the issue
- Fallbacks that hide issues — just crash
- Wildcard re-exports (`export * from`) — use named exports
- Legacy/backward compatibility shims — delete the old path, move forward
- Raw-dogging APIs — use official SDKs when they exist
- Hardcoding what should be discovered — let the flow find it
- Improvising what should be in the skill — every improvisation is a skill gap

### Package Rules (`packages/agent`, `packages/bench`, `packages/cdp`)

- Prefer explicit, caller-controlled configuration for package-owned behavior
- Do not introduce hidden dependencies on ambient shell state for defaults
- Do not change inherited environment semantics accidentally; if env policy changes, make it a deliberate API decision
- Prefer deterministic defaults and fail fast when required configuration is missing
- Do not guess from `cwd`, `PATH`, or repository layout when a package-owned path or config contract can be used
- Preserve the caller's `cwd` semantics unless changing it is the explicit purpose of the operation

### Zero-Lint Policy

All linters must pass with zero errors and zero warnings:

```bash
bun run ci  # typecheck + lint + test + knip
```

---

## Guidelines (Water — Adapt)

### LLM Decision Protocol

**Never make architectural changes without user consultation.**

Requires approval: Module formats, build tools, deployment strategies, core architecture.

When uncertain: Stop, explain problem + options, get approval, execute.

### Code Style

- **Imports**: Inline type specifiers (`import { type Foo, bar }`), no duplicates, members sorted within statement.
- **Exports**: Named exports only. No barrel wildcard re-exports. Remove unused exports (knip enforces this).
- **Function size**: Target ~10 statements per function. Extract when complexity grows.
- **Large modules**: When a module grows to multiple responsibilities, turn it into a folder. Keep the public entry file small and split internals by concern instead of growing a monolithic file.
- **Object keys**: Prefer semantic grouping over alphabetical.

### Soft Lint (Water — Guidance, Not Enforcement)

- **Magic numbers**: Extract named constants for non-obvious values. `0`, `1`, `-1` are fine. Timeouts, array sizes, thresholds → name them.
- **Positive conditions**: Prefer `if (ready)` over `if (!notReady)`. Negate only when the negative path is the main logic.
- **Function style**: Module-level exports → function declarations (hoisting, stack traces). Callbacks → arrow functions.

---

## Workflow Standards

### Project Context

- Never run scripts directly — always `bun run <script>` via package.json
- Package tests run from the monorepo root: `bun run test`
- Root `tsconfig.json` has path mappings for workspace packages
- Read the code before acting — the code is the source of truth, not your mental model

### Git

- Push immediately after work
- Before push: check for documentation drift (README.md, CLAUDE.md, AGENTS.md)
- **Never enable auto-merge** unless user explicitly requests it

### Linear Tracking (Non-Negotiable)

- **Every PR must have a Linear ticket.** No exceptions. Create the ticket before or at PR time.
- Branch names must include the Linear issue ID (e.g., `feat/ENG-123-add-widget`).
- PR title/description must reference the Linear issue so it auto-links.
- If no ticket exists for the work, create one in Linear first — leadership reporting depends on this.

### PR Discipline: One Thought Per PR

- Each PR addresses exactly one concern. One bug, one feature, one refactor.
- If you discover an unrelated issue while working, **stop** — create a separate Linear ticket and a separate branch/PR for it. Do not bundle.
- Mixing concerns makes PRs unreadable and breaks Linear tracking granularity.
- When in doubt, split. Small PRs merge fast; large PRs rot.

### Documentation

- **TSDoc** — Function/field level documentation for all exports
- **Comments** — Explain non-trivial logic; quality code is documented code
- **Doc conflicts** — Use `git log`/`git blame` to find newest statements; newest wins; remove stale claims

---

## Toolchain

| Tool     | Purpose                          | Command                            |
| -------- | -------------------------------- | ---------------------------------- |
| oxlint   | Lint (with `--deny-warnings`)    | `bun run lint`                     |
| tsc      | Typecheck (`--noEmit`)           | `bun run typecheck`                |
| knip     | Dead export / unused dep         | `bun run knip`                     |
| bun test | Test runner                      | `bun run test`                     |
| lefthook | Git hooks (pre-commit, pre-push) | automatic                          |

---

## References

| Document | What it provides |
| -------- | ---------------- |
| [Block as Intelligence](https://docs.google.com/document/d/171-gwd5yqnxHfuMAdFAU3l9SuIKOzjxiXMCVYW6GchA/edit?tab=t.0) | The operating model — world models, proactive intelligence, capabilities, interfaces, org architecture. Four layers (state, causal, intent, prediction). The flywheel. Loss functions over KPIs. |

## Archetypes (Elements)

- **Steve Jobs (Earth)**: Build solid ground. Break it only to raise higher—never just to dig holes.
- **Bruce Lee (Water)**: Be formless. Adapt to any shape. Flow smooth or crash hard—both are water.
- **Jiro Ono (Fire)**: Question every ingredient. Delete the excess. Simplify what remains. Accelerate the cycle. Automate last—only after the blade is sharp.
- **David Bowie (Wind)**: Observe the shift before it happens. Position early. Reinvent when the pattern breaks.
- **John Wick (Void)**: Silence after the bullet. No echo, no corpse, no grief. Kill the branch. Commit nothing. Walk away. The trunk breathes again.

Channel them, but never lose the blade. These aren't options. They're tempering. Use one if the code sings louder. But every line returns to Musashi. The repo isn't an art gallery—it's a duel. Pick fast or don't. No politics. Just steel.
