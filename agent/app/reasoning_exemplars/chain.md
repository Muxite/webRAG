# Reasoning exemplar: sequential dependency chain

A chain task gives you a terminal fact to find, reachable only by hopping
page-to-page where each hop's answer names the next page. You cannot plan past
hop 1; each next target is revealed, not guessed. Read pages — never answer a hop
from memory.

Situation: The prompt uniquely describes the entity of step 1 (a distinctive
attribute plus a qualifier). Nothing downstream is known yet.
Thought: Step 1's target is directly identifiable from the prompt alone — it is the
only hop I can resolve up front. I don't yet hold the fact the next hop needs.
Action: Search for that description; identify the step-1 entity by name.

Situation: I have the entity from step 1, but not the detail that points to the
next page.
Thought: The next target isn't guessable in advance — it is only knowable from what
the step-1 entity's own page reveals. So I go to that page and read the specific
detail the chain hangs on. I also watch for a qualifying detail (a country, a date,
a category) that will matter for disambiguation later.
Action: Open the step-1 entity's page; extract both the linking detail AND any
qualifier attached to it.

Situation: The linking detail names a place/thing, but a bare name can match
several same-named decoys.
Thought: I must carry the qualifying detail from the prior page forward into this
lookup, or I risk landing on a wrong homonym page. The target's identity =
name + qualifier, not name alone.
Action: Open the page for (name + carried qualifier). Confirm the page matches the
qualifier before trusting it.

Situation: The current page is the one the prompt said holds the terminal fact, in
a named location (e.g. an infobox / a specific field).
Thought: The requested fact is present right here, in the location the prompt
specified. No further hop is required — chasing another link would overshoot.
Action: Read the terminal fact from that exact location. Report it plus the full
chain and every URL read. Done.
