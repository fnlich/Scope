Solve this programming problem in Rust.

Rules — the grader is automated and unforgiving:
- Reply with ONLY ONE Rust code block and nothing else outside it.
- Write ONE complete program with `fn main()`, compiled as a single file with
  `rustc --edition=2021 -C opt-level=2`. No Cargo, no crates, std only.
- READ the input from stdin and WRITE only the requested answer to stdout.
- Output is compared token-by-token after splitting on ASCII whitespace, so
  extra prose, labels or prompts make the answer wrong.
- Write no comments and no docstrings. Nothing reads them, and the answer is
  scored partly on how fast it arrives.

Edge cases are where this is won or lost. Walk your solution through every one
of these before you answer, and fix what breaks:
- NOTHING: an empty list, an empty string, n = 0. Work out what the statement
  says the answer is, then make sure your code reaches it instead of crashing
  or dividing by a length of zero.
- ONE: n = 1, a single element, a one-character string. Almost every
  off-by-one bug is visible here and nowhere else.
- TWO: the smallest case where "first" and "last" are different elements, and
  where adjacent-pair logic stops agreeing with the general case.
- BOTH ENDS: the first element and the last, an empty range, and an inclusive
  bound against an exclusive one. Check the far end explicitly — a loop that is
  right at the start and short by one at the finish passes every example.
- EXTREME VALUES: 0, 1, -1, negative numbers, and the largest magnitude the
  statement allows. If no bound is given, assume values up to 10^18 and n up to
  10^5, and pick an algorithm and a type that survive both.
- DEGENERATE SHAPE: every element equal, every element a duplicate, already
  sorted, sorted backwards, all zeros.
- INTEGER OVERFLOW IS SILENT HERE. The grader compiles with `-C opt-level=2`,
  which turns overflow checks OFF: `i32` arithmetic wraps around and the
  program exits normally with a wrong answer instead of panicking. Two `i32`
  values of 2_000_000_000 add up to -294967296. Use `i64` everywhere by
  default, `i128` for products, and reach for `i32` only where you have proved
  the range cannot be exceeded.
- Read to EOF. Tolerate trailing newlines, blank lines and repeated spaces, and
  do not assume a fixed number of lines unless the statement fixes it.
- Deep recursion overflows the stack. Prefer iteration for n up to 10^5.
- Each test gets about 5 seconds, so lock stdout once and wrap it in a
  BufWriter rather than printing in a loop.

PROBLEM:
Summary: Bind ambiguous imported bones under hierarchy constraints

Implement a complete Rust program containing `fn main()` that deterministically binds bone records to nodes of a rooted imported scene. Names are not unique. A bone may bind only to a scene node with the same name, and each scene node may be used at most once.

Input:
- `N B`
- `N` lines describing scene nodes `0..N-1`: `parent name`. Node `0` has parent `-1`; every other parent is in `0..i-1`.
- `B` lines describing bones `0..B-1`: `parent name`. Each parent is `-1` or in `0..i-1`.

A scene node is a strict ancestor of another if it occurs above it in the scene tree.

At any moment, a matching unused scene node is an admissible candidate for an unresolved bone exactly when both conditions hold:
- If the bone's parent is resolved, that parent's assigned scene node is a strict ancestor of the candidate.
- For every resolved direct child of the bone, the candidate is a strict ancestor of that child's assigned scene node.

Initially all bones are unresolved and all scene nodes are unused. Repeatedly perform exactly one transition:

1. Singleton transition: If any unresolved bone has exactly one admissible candidate, choose the smallest-ID such bone and bind it to that candidate.

2. Relational transition: Used only when no singleton transition exists. For every unresolved bone and each of its admissible candidates, compute `(P,C,S)`:
   - `P=1` iff the bone's parent is resolved and its assigned node is the candidate's direct scene parent; otherwise `0`.
   - `C` is the number of resolved direct children whose assigned nodes have the candidate as direct scene parent.
   - `S` is the number of resolved siblings whose assigned nodes have the same direct scene parent as the candidate. A bone with parent `-1` has no siblings.

For one bone, a candidate is eligible only if its tuple is lexicographically greatest among that bone's admissible candidates, uniquely attains that tuple, and is not `(0,0,0)`. If eligible candidates exist, choose the one having the lexicographically greatest tuple, breaking ties by smaller bone ID, and bind it.

3. If neither transition applies, stop. Remaining bones are unresolved.

Admissibility, counts, and choices are recomputed after every binding.

Names are nonempty ASCII alphanumeric strings. `1 <= N,B <= 200000`. Over all names, the sum of `(scene-node count) * (bone count)` is at most `1000000`. Over every ordered pair of distinct bones `(x,y)` where `y` is the parent, a child, or a non-root sibling of `x`, the sum of the number of scene nodes sharing `x`'s name is at most `3000000`.

Output `B` integers in bone-ID order: the assigned scene-node ID, or `-1` if unresolved. Judging splits output only on ASCII whitespace bytes and compares all tokens exactly.

Use only the Rust standard library. Do not use unsafe code, external crates, filesystem access, network access, or nondeterminism.

Reply directly in the chat with one ordinary fenced code block, however long the program is. Do not use canvas.