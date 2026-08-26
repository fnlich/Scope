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
Given finite sets `A={1..N}`, `B={1..M}`, and `C={1..R}`, each element has an explicit equivalence-class label. Two elements of the same set are equivalent exactly when their labels are equal. In each set, labels are consecutive starting at `1`, and every stated class occurs.

A table `f` maps `A` to `B`, and a table `g` maps `B` to `C`. The input guarantees:

- equivalent `A` elements map under `f` to equivalent `B` elements;
- equivalent `B` elements map under `g` to equivalent `C` elements;
- every `B` class is reached by `f` modulo equivalence;
- every `C` class is reached by `g` modulo equivalence.

For every `C` class `i`, in increasing label order, select one witness `w[i]` from `A` whose composite image `g[f[w[i]]]` belongs to class `i`.

The witness table must satisfy, for every `2 <= i <= KC`:

- `|w[i]-w[i-1]| <= X`;
- `|f[w[i]]-f[w[i-1]]| <= Y`.

It must also satisfy the global condition

`(classA[w[1]] + classA[w[2]] + ... + classA[w[KC]]) mod 2 = P`.

Among all valid witness tables, minimize `max(w[i])`. Subject to that, output the lexicographically smallest sequence `w[1],...,w[KC]`. If none exists, output `IMPOSSIBLE`.

Input consists of ASCII-whitespace-separated integers:

`N M R KA KB KC X Y P`

followed by five tables, which may span arbitrary lines:

- `N` labels `classA[a]` in `1..KA`;
- `M` labels `classB[b]` in `1..KB`;
- `R` labels `classC[c]` in `1..KC`;
- `N` values `f[a]` in `1..M`;
- `M` values `g[b]` in `1..R`.

Constraints:

- `1 <= N,M,R <= 200000`;
- `1 <= KA <= N`, `1 <= KB <= M`, `1 <= KC <= R`;
- `0 <= X <= N`, `0 <= Y <= M`;
- `P` is `0` or `1`;
- all guarantees above hold.

Output either the single token `IMPOSSIBLE`, or the minimum possible maximum followed by the `KC` selected source indices.

Judging splits output only on ASCII whitespace bytes `0x09` through `0x0D` and `0x20`, then compares tokens exactly. Provide one deterministic Rust program using only the standard library, containing `fn main()`.

Reply directly in the chat with one ordinary fenced code block, however long the program is. Do not use canvas.