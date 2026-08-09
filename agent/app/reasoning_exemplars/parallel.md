# Reasoning exemplar: independent parallel chains merged by computation

Use this pattern when a prompt asks you to gather two (or more) facts by
following separate lookup chains and then combine them with an operation.

**Situation:** The prompt decomposes into 2+ sub-goals (A, B, ...). Each
sub-goal is its own multi-hop chain, and no chain needs another chain's result
to proceed. A final combining operation (difference, sum, ratio, comparison,
concatenation, ...) joins their terminal facts into one answer.

**Thought (decompose):** These sub-chains are independent, so I can pursue them
in parallel — I don't have to finish A before starting B. But the task is NOT
done when any single chain finishes; the answer needs EVERY chain's terminal
fact fed into the combining operation.

**Action — sub-goal A, hop 1:** Resolve A's first target. The hop-2 target is
only knowable from what A's hop-1 page actually says, so read it; don't guess.

**Action — sub-goal B, hop 1:** Independently resolve B's first target from B's
own hop-1 page.

**Thought (advance each chain):** Each chain walks hop-by-hop to its own
terminal fact, exactly as a lone chain would. Each hop's next target comes only
from the previous hop's page.

**Action — sub-goal A, hop 2:** Read A's hop-2 page; extract A's terminal fact.

**Action — sub-goal B, hop 2:** Read B's hop-2 page; extract B's terminal fact.

**Thought (check completeness):** I now hold a terminal fact for every
sub-goal. Only now is the task ready to finish — a single chain alone would be a
partial, wrong answer.

**Action — merge:** Apply exactly the combining operation the prompt named to
the terminal facts, producing the final result.

**Report:** State the computed final result (the keystone) AND each chain's raw
intermediate facts with the exact URL of every page read — the intermediates
give verifiability and partial credit, the computed value is the deliverable.
Never report only one or the other.
