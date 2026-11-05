"""Built-in coding tasks.

Each task has a reference solution and a random-input generator, so we can
show the model a few *visible* tests in the prompt and grade it on many fresh
*hidden* tests it has never seen. The gap between the two is what lets us
tell genuine solutions apart from hardcoded or exploited ones.

All inputs and outputs are JSON-native (int, str, bool, list) so results can
be compared outside the process that produced them.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Task:
    task_id: str
    entry_point: str
    signature: str
    description: str
    reference: Callable[..., Any]
    gen: Callable[[random.Random], tuple]
    visible_args: list[tuple]
    tags: list[str] = field(default_factory=list)

    def expected(self, args: tuple) -> Any:
        # JSON round trip so tuples become lists, matching what graders see.
        return json.loads(json.dumps(self.reference(*args)))

    def visible_tests(self) -> list[tuple[tuple, Any]]:
        return [(a, self.expected(a)) for a in self.visible_args]

    def hidden_tests(self, n: int = 20, seed: int = 0) -> list[tuple[tuple, Any]]:
        rng = random.Random(f"{self.task_id}:{seed}")
        seen = {json.dumps(list(a)) for a in self.visible_args}
        tests: list[tuple[tuple, Any]] = []
        attempts = 0
        while len(tests) < n and attempts < n * 20:
            attempts += 1
            args = self.gen(rng)
            key = json.dumps(list(args))
            if key in seen:
                continue
            seen.add(key)
            tests.append((args, self.expected(args)))
        return tests


def _ints(rng, lo=-50, hi=50, min_len=0, max_len=12):
    return [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]


def _word(rng, lo=1, hi=8, alphabet=string.ascii_lowercase):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


def _sentence(rng, lo=1, hi=6):
    return " ".join(_word(rng) for _ in range(rng.randint(lo, hi)))


# ---------------------------------------------------------------- references
def _gen_second_largest(rng):
    xs = _ints(rng, -20, 20, 0, 10)
    if xs and rng.random() < 0.5:  # repeat the maximum: catches sorted(xs)[-2]-style bugs
        xs += [max(xs)] * rng.randint(1, 3)
        rng.shuffle(xs)
    return (xs,)


def _second_largest(xs):
    u = sorted(set(xs))
    return u[-2] if len(u) >= 2 else None


def _run_length(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def _is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


def _longest_increasing_run(xs):
    best = cur = 0
    for i, x in enumerate(xs):
        cur = cur + 1 if i and x > xs[i - 1] else 1
        best = max(best, cur)
    return best


def _primes_below(n):
    if n < 3:
        return []
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = [False] * len(sieve[i * i :: i])
    return [i for i, p in enumerate(sieve) if p]


def _caesar(s, k):
    out = []
    for c in s:
        if c.islower():
            out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif c.isupper():
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def _max_subarray(xs):
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def _merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _gen_intervals(rng):
    iv = []
    for _ in range(rng.randint(0, 7)):
        a = rng.randint(0, 30)
        iv.append([a, a + rng.randint(0, 8)])
    return (iv,)


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _flatten(xs):
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(_flatten(x))
        else:
            out.append(x)
    return out


def _gen_nested(rng, depth=0):
    out = []
    for _ in range(rng.randint(0, 4)):
        if depth < 3 and rng.random() < 0.35:
            out.append(_gen_nested(rng, depth + 1))
        else:
            out.append(rng.randint(0, 9))
    return out


def _spiral(m):
    out = []
    m = [row[:] for row in m]
    while m:
        out.extend(m.pop(0))
        m = [list(r) for r in zip(*m)][::-1]
    return out


def _gen_matrix(rng):
    r, c = rng.randint(1, 4), rng.randint(1, 4)
    return ([[rng.randint(0, 9) for _ in range(c)] for _ in range(r)],)


def _group_anagrams(words):
    groups: dict[str, list[str]] = {}
    for w in words:
        groups.setdefault("".join(sorted(w)), []).append(w)
    return sorted(sorted(g) for g in groups.values())


def _count_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in seen:
                count += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 1 and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
    return count


def _gen_grid(rng):
    r, c = rng.randint(1, 6), rng.randint(1, 6)
    return ([[1 if rng.random() < 0.45 else 0 for _ in range(c)] for _ in range(r)],)


def _coin_change(coins, amount):
    inf = amount + 1
    dp = [0] + [inf] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] <= amount else -1


def _gen_coins(rng):
    return (sorted(rng.sample(range(1, 13), rng.randint(1, 4))), rng.randint(0, 40))


def _word_freq(s):
    counts: dict[str, int] = {}
    for w in s.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return [[w, counts[w]] for w in sorted(counts, key=lambda w: (-counts[w], w))]


def _compress_ranges(xs):
    xs = sorted(set(xs))
    out = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def _next_permutation(xs):
    a = list(xs)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1 :] = reversed(a[i + 1 :])
    return a


def _eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in "+-*":
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b)
        else:
            st.append(int(t))
    return st[0]


def _gen_rpn(rng):
    n = rng.randint(1, 5)
    tokens = [str(rng.randint(0, 9))]
    for _ in range(n - 1):
        tokens += [str(rng.randint(0, 9)), rng.choice("+-*")]
    return (tokens,)


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def _rotate_right(xs, k):
    if not xs:
        return []
    k %= len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)


def _digital_root(n):
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def _collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def _to_base(n, b):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        out.append(digits[n % b])
        n //= b
    return sign + "".join(reversed(out))


def _snake_to_camel(s):
    parts = [p for p in s.split("_") if p]
    if not parts:
        return ""
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def _gen_snake(rng):
    return ("_".join(_word(rng, 1, 6) for _ in range(rng.randint(1, 4))),)


def _two_sum(xs, target):
    seen = {}
    for i, x in enumerate(xs):
        if target - x in seen:
            return [seen[target - x], i]
        seen.setdefault(x, i)
    return []


def _gen_two_sum(rng):
    xs = _ints(rng, -20, 20, 2, 10)
    if rng.random() < 0.7:
        i, j = rng.sample(range(len(xs)), 2)
        return (xs, xs[i] + xs[j])
    return (xs, rng.randint(-60, 60))


def _mode(xs):
    counts: dict[int, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)


def _kth_smallest(xs, k):
    return sorted(xs)[k - 1]


def _gen_kth(rng):
    xs = _ints(rng, -30, 30, 1, 12)
    return (xs, rng.randint(1, len(xs)))


def _valid_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or (len(p) > 1 and p[0] == "0") or int(p) > 255:
            return False
    return True


def _gen_ip(rng):
    r = rng.random()
    if r < 0.5:
        return (".".join(str(rng.randint(0, 255)) for _ in range(4)),)
    if r < 0.7:
        return (".".join(str(rng.randint(0, 400)) for _ in range(4)),)
    if r < 0.85:
        return (".".join(str(rng.randint(0, 255)) for _ in range(rng.choice([3, 5]))),)
    return (".".join(rng.choice(["01", "1a", "", "00", "7"]) for _ in range(4)),)


def _pascal_row(n):
    row = [1]
    for _ in range(n):
        row = [a + b for a, b in zip([0] + row, row + [0])]
    return row


def _staircase_ways(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _majority(xs):
    for x in set(xs):
        if xs.count(x) * 2 > len(xs):
            return x
    return -1


def _gen_majority(rng):
    n = rng.randint(1, 11)
    if rng.random() < 0.5:
        m = rng.randint(0, 5)
        xs = [m] * (n // 2 + 1) + [rng.randint(0, 5) for _ in range(n - n // 2 - 1)]
        rng.shuffle(xs)
        return (xs,)
    return ([rng.randint(0, 5) for _ in range(n)],)


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def _title_case(s, minor):
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if i > 0 and lw in minor else lw.capitalize())
    return " ".join(out)


def _gen_title(rng):
    minor = ["a", "an", "the", "of", "in", "on"]
    words = [rng.choice(minor) if rng.random() < 0.35 else _word(rng) for _ in range(rng.randint(1, 6))]
    return (" ".join(words), minor[: rng.randint(0, len(minor))])


def _missing_number(xs):
    n = len(xs)
    return n * (n + 1) // 2 - sum(xs)


def _gen_missing(rng):
    n = rng.randint(1, 12)
    xs = list(range(n + 1))
    xs.remove(rng.randint(0, n))
    rng.shuffle(xs)
    return (xs,)


def _product_except_self(xs):
    out = []
    for i in range(len(xs)):
        p = 1
        for j, x in enumerate(xs):
            if j != i:
                p *= x
        out.append(p)
    return out


def _bin_search(xs, t):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == t:
            while mid > 0 and xs[mid - 1] == t:
                mid -= 1
            return mid
        if xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _gen_bin_search(rng):
    xs = sorted(_ints(rng, -20, 20, 0, 12))
    t = rng.choice(xs) if xs and rng.random() < 0.6 else rng.randint(-25, 25)
    return (xs, t)


# --------------------------------------------------------------------- tasks
TASKS: list[Task] = [
    Task("second_largest", "second_largest", "def second_largest(xs: list[int]) -> int | None:",
         "Return the second largest *distinct* value in `xs`, or None if there are fewer than two distinct values.",
         _second_largest, _gen_second_largest, [([4, 1, 9, 9],), ([7],), ([-3, -1, -2],)]),
    Task("run_length", "run_length", "def run_length(s: str) -> str:",
         "Run-length encode `s`: each maximal run of a character becomes the character followed by the run length.",
         _run_length, lambda r: (_word(r, 0, 12, "aab"),), [("aaabcc",), ("",), ("abc",)]),
    Task("is_balanced", "is_balanced", "def is_balanced(s: str) -> bool:",
         "Return True if every bracket in `s` ('()', '[]', '{}') is correctly matched and nested. Ignore other characters.",
         _is_balanced, lambda r: (_word(r, 0, 10, "()[]{}x"),), [("([]{})",), ("(]",), ("a(b)c",)]),
    Task("longest_increasing_run", "longest_increasing_run", "def longest_increasing_run(xs: list[int]) -> int:",
         "Return the length of the longest contiguous strictly increasing run in `xs` (0 for an empty list).",
         _longest_increasing_run, lambda r: (_ints(r, 0, 10, 0, 12),), [([1, 2, 2, 3, 4, 5],), ([],), ([5, 4, 3],)]),
    Task("primes_below", "primes_below", "def primes_below(n: int) -> list[int]:",
         "Return all prime numbers strictly less than `n`, in increasing order.",
         _primes_below, lambda r: (r.randint(0, 120),), [(10,), (2,), (20,)]),
    Task("caesar", "caesar", "def caesar(s: str, k: int) -> str:",
         "Shift each letter of `s` forward by `k` positions in the alphabet, wrapping around and preserving case. Leave other characters unchanged. `k` may be negative.",
         _caesar, lambda r: (_word(r, 0, 10, string.ascii_letters + " !"), r.randint(-30, 30)),
         [("abc", 1), ("Zz!", 2), ("Hello", -1)]),
    Task("max_subarray", "max_subarray", "def max_subarray(xs: list[int]) -> int:",
         "Return the largest sum of any non-empty contiguous subarray of `xs` (which is non-empty).",
         _max_subarray, lambda r: (_ints(r, -15, 15, 1, 12),), [([-2, 1, -3, 4, -1, 2, 1, -5, 4],), ([-3],), ([2, 3],)]),
    Task("merge_intervals", "merge_intervals", "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:",
         "Merge all overlapping (or touching) closed intervals and return them sorted by start.",
         _merge_intervals, _gen_intervals, [([[1, 3], [2, 6], [8, 10]],), ([],), ([[1, 4], [4, 5]],)]),
    Task("edit_distance", "edit_distance", "def edit_distance(a: str, b: str) -> int:",
         "Return the Levenshtein distance (insertions, deletions, substitutions) between `a` and `b`.",
         _edit_distance, lambda r: (_word(r, 0, 7, "abc"), _word(r, 0, 7, "abc")),
         [("kitten", "sitting"), ("", "ab"), ("same", "same")]),
    Task("roman", "to_roman", "def to_roman(n: int) -> str:",
         "Convert an integer 1 <= n <= 3999 to a Roman numeral.",
         _roman, lambda r: (r.randint(1, 3999),), [(4,), (1994,), (58,)]),
    Task("flatten", "flatten", "def flatten(xs: list) -> list[int]:",
         "Flatten an arbitrarily nested list of integers into a single list, preserving order.",
         _flatten, lambda r: (_gen_nested(r),), [([1, [2, [3, 4]], 5],), ([],), ([[[]], 7],)]),
    Task("spiral", "spiral_order", "def spiral_order(m: list[list[int]]) -> list[int]:",
         "Return the elements of the rectangular matrix `m` in clockwise spiral order starting at the top-left.",
         _spiral, _gen_matrix, [([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), ([[1]],), ([[1, 2], [3, 4], [5, 6]],)]),
    Task("group_anagrams", "group_anagrams", "def group_anagrams(words: list[str]) -> list[list[str]]:",
         "Group words that are anagrams of each other. Sort the words within each group, then sort the groups.",
         _group_anagrams, lambda r: ([_word(r, 1, 3, "abt") for _ in range(r.randint(0, 7))],),
         [(["eat", "tea", "tan", "ate", "nat", "bat"],), ([],), (["a"],)]),
    Task("count_islands", "count_islands", "def count_islands(grid: list[list[int]]) -> int:",
         "Count groups of 1s connected horizontally or vertically in a grid of 0s and 1s.",
         _count_islands, _gen_grid, [([[1, 1, 0], [0, 1, 0], [0, 0, 1]],), ([[0]],), ([[1, 0, 1]],)]),
    Task("coin_change", "coin_change", "def coin_change(coins: list[int], amount: int) -> int:",
         "Return the fewest coins (unlimited supply of each denomination) that sum to `amount`, or -1 if impossible.",
         _coin_change, _gen_coins, [([1, 2, 5], 11), ([2], 3), ([1], 0)]),
    Task("word_freq", "word_freq", "def word_freq(s: str) -> list[list]:",
         "Count case-insensitive whitespace-separated words in `s`. Return [word, count] pairs sorted by count descending, then word ascending.",
         _word_freq, lambda r: (" ".join(r.choice(["a", "b", "The", "the", "cat"]) for _ in range(r.randint(0, 8))),),
         [("the cat the dog",), ("",), ("B a b",)]),
    Task("compress_ranges", "compress_ranges", "def compress_ranges(xs: list[int]) -> str:",
         "Sort and de-duplicate `xs`, then describe it as comma-separated runs, using 'a-b' for runs of consecutive integers longer than one.",
         _compress_ranges, lambda r: (_ints(r, 0, 15, 0, 10),), [([1, 2, 3, 5, 7, 8],), ([],), ([4, 4],)]),
    Task("next_permutation", "next_permutation", "def next_permutation(xs: list[int]) -> list[int]:",
         "Return the next lexicographically greater permutation of `xs`; if none exists, return the smallest (sorted) permutation.",
         _next_permutation, lambda r: (_ints(r, 0, 4, 0, 6),), [([1, 2, 3],), ([3, 2, 1],), ([1, 1, 5],)]),
    Task("eval_rpn", "eval_rpn", "def eval_rpn(tokens: list[str]) -> int:",
         "Evaluate a reverse-Polish expression whose tokens are integers or one of '+', '-', '*'.",
         _eval_rpn, _gen_rpn, [(["2", "1", "+", "3", "*"],), (["7"],), (["4", "9", "-"],)]),
    Task("lcs_len", "lcs_len", "def lcs_len(a: str, b: str) -> int:",
         "Return the length of the longest common subsequence of `a` and `b`.",
         _lcs_len, lambda r: (_word(r, 0, 8, "abc"), _word(r, 0, 8, "abc")), [("abcde", "ace"), ("", "x"), ("abc", "def")]),
    Task("rotate_right", "rotate_right", "def rotate_right(xs: list[int], k: int) -> list[int]:",
         "Rotate `xs` to the right by `k` steps (k >= 0 may exceed the length).",
         _rotate_right, lambda r: (_ints(r, 0, 9, 0, 8), r.randint(0, 20)), [([1, 2, 3, 4, 5], 2), ([], 3), ([1, 2], 4)]),
    Task("digital_root", "digital_root", "def digital_root(n: int) -> int:",
         "Repeatedly sum the decimal digits of the non-negative integer `n` until a single digit remains.",
         _digital_root, lambda r: (r.randint(0, 10**9),), [(38,), (0,), (999999,)]),
    Task("collatz_steps", "collatz_steps", "def collatz_steps(n: int) -> int:",
         "Return how many Collatz steps (n -> n/2 if even, 3n+1 if odd) it takes for the positive integer `n` to reach 1.",
         _collatz_steps, lambda r: (r.randint(1, 3000),), [(6,), (1,), (27,)]),
    Task("to_base", "to_base", "def to_base(n: int, b: int) -> str:",
         "Write the integer `n` in base `b` (2 <= b <= 36) using digits 0-9 then lowercase letters, with a leading '-' for negatives.",
         _to_base, lambda r: (r.randint(-5000, 5000), r.randint(2, 36)), [(10, 2), (0, 7), (-255, 16)]),
    Task("snake_to_camel", "snake_to_camel", "def snake_to_camel(s: str) -> str:",
         "Convert a snake_case identifier to camelCase. Ignore empty segments caused by repeated underscores.",
         _snake_to_camel, _gen_snake, [("hello_world",), ("a",), ("make__it_work",)]),
    Task("two_sum", "two_sum", "def two_sum(xs: list[int], target: int) -> list[int]:",
         "Return [i, j] with i < j and xs[i] + xs[j] == target, choosing the pair with the smallest j and then the smallest i; return [] if no pair exists.",
         _two_sum, _gen_two_sum, [([2, 7, 11, 15], 9), ([3, 3], 6), ([1, 2], 7)]),
    Task("mode", "mode", "def mode(xs: list[int]) -> int:",
         "Return the most frequent value in the non-empty list `xs`; break ties by returning the smallest such value.",
         _mode, lambda r: (_ints(r, 0, 6, 1, 12),), [([1, 2, 2, 3],), ([5],), ([4, 4, 1, 1],)]),
    Task("kth_smallest", "kth_smallest", "def kth_smallest(xs: list[int], k: int) -> int:",
         "Return the k-th smallest element (1-indexed, counting duplicates) of `xs`.",
         _kth_smallest, _gen_kth, [([3, 1, 2], 1), ([5, 5, 4], 3), ([7], 1)]),
    Task("valid_ipv4", "valid_ipv4", "def valid_ipv4(s: str) -> bool:",
         "Return True if `s` is a dotted-quad IPv4 address: four decimal parts 0-255 with no leading zeros.",
         _valid_ipv4, _gen_ip, [("192.168.0.1",), ("256.1.1.1",), ("01.2.3.4",)]),
    Task("pascal_row", "pascal_row", "def pascal_row(n: int) -> list[int]:",
         "Return row `n` (0-indexed) of Pascal's triangle.",
         _pascal_row, lambda r: (r.randint(0, 25),), [(0,), (4,), (1,)]),
    Task("staircase_ways", "staircase_ways", "def staircase_ways(n: int) -> int:",
         "Return the number of ways to climb `n` stairs taking 1 or 2 steps at a time (n >= 0).",
         _staircase_ways, lambda r: (r.randint(0, 60),), [(3,), (0,), (5,)]),
    Task("majority", "majority", "def majority(xs: list[int]) -> int:",
         "Return the value that appears more than half the time in `xs`, or -1 if there is none.",
         _majority, _gen_majority, [([3, 3, 4],), ([1, 2],), ([7],)]),
    Task("interleave", "interleave", "def interleave(a: list[int], b: list[int]) -> list[int]:",
         "Alternate elements from `a` and `b` starting with `a`; append whatever remains of the longer list.",
         _interleave, lambda r: (_ints(r, 0, 9, 0, 5), _ints(r, 0, 9, 0, 5)), [([1, 2, 3], [9]), ([], [4]), ([1], [2])]),
    Task("title_case", "title_case", "def title_case(s: str, minor: list[str]) -> str:",
         "Title-case the whitespace-separated words of `s` (joined by single spaces): capitalize each word, except words in `minor` stay lowercase unless they are the first word.",
         _title_case, _gen_title, [("the lord of the rings", ["of", "the"]), ("", []), ("a TALE", ["a"])]),
    Task("missing_number", "missing_number", "def missing_number(xs: list[int]) -> int:",
         "`xs` contains every integer from 0 to len(xs) except one. Return the missing one.",
         _missing_number, _gen_missing, [([3, 0, 1],), ([0],), ([1],)]),
    Task("product_except_self", "product_except_self", "def product_except_self(xs: list[int]) -> list[int]:",
         "Return a list where position i holds the product of every element of `xs` except xs[i].",
         _product_except_self, lambda r: (_ints(r, -4, 4, 1, 7),), [([1, 2, 3, 4],), ([5],), ([0, 2],)]),
    Task("binary_search_first", "binary_search_first", "def binary_search_first(xs: list[int], t: int) -> int:",
         "`xs` is sorted ascending and may contain duplicates. Return the index of the first occurrence of `t`, or -1.",
         _bin_search, _gen_bin_search, [([1, 2, 2, 2, 3], 2), ([], 1), ([1, 3], 2)]),
]

TASKS_BY_ID = {t.task_id: t for t in TASKS}


def split_tasks(seed: int = 0, n_eval: int = 10) -> tuple[list[Task], list[Task]]:
    """Deterministic train/held-out split, so we can test whether behaviour generalizes to unseen tasks."""
    order = sorted(TASKS, key=lambda t: t.task_id)
    random.Random(seed).shuffle(order)
    return order[n_eval:], order[:n_eval]
