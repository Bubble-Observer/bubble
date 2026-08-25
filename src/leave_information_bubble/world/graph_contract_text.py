"""One domain-neutral graph cognition contract shared by prompts and tools.

The durable world model speaks one vocabulary: objects (six kinds), events as
time anchors, assertions, observations, inquiries, normalized type keys,
bounded qualifiers, identity aliases, and name usage. The agent prompts and
the memory tool descriptions all reference this text (or its identity
sentence) instead of re-defining the rules, so the model and the
compile-time validators talk about the same world. The historical terminal
``submit_cognition`` surface keeps its own narrower three-kind schema; this
contract describes the active graph_patch write path.

``render_contract_text()`` returns the short contract text and
``render_contract_examples()`` the six domain-neutral write-shape example
groups. Both are stable pure strings with no world-module dependencies; the
normalization rule bodies themselves stay in ``graph_contract.py``.
"""

from __future__ import annotations

IDENTITY_MODEL_SENTENCE = (
    "A canonical name is the default display name and nothing more, never a "
    "unique identity. A legacy name is history, never an identity key. Only an "
    "active identity alias is world-unique; aliases are normalized before "
    "comparison. Name usage is an explicit assertion that may link supporting "
    "evidence when available; "
    "the host never auto-creates identity claims."
)

EVENT_REIFICATION_RULE = (
    "Before staging a direct object-to-object assertion, decide what the "
    "relationship itself denotes. A relation that remains meaningful without "
    "one particular occurrence or time (such as membership, ownership, or "
    "location) is a direct assertion. A bounded occurrence or outcome that "
    "connects multiple entities, or marks a change of state in one entity, is "
    "an event object when it is worth preserving and its instant, day, or "
    "interval is known: connect the event to every known participant or "
    "changed subject, and attach occurrence-specific results and details to "
    "the event. "
    "This topology rule does not depend on importance: omit a low-value "
    "occurrence if it is not worth preserving, but never compress a preserved "
    "occurrence into a direct edge between its participants. A containing "
    "episode or period does not replace a preserved child occurrence; either "
    "create the child event and link it to the container, or omit the "
    "child-specific judgment. Qualifiers only refine one assertion edge; never "
    "use role, scope, or another qualifier to carry the occurrence's date, "
    "result, or participant list instead of an event. The host cannot infer a "
    "missing event from a direct edge."
)

_CONTRACT_BODY = f"""An object is a durable node; an assertion is revisable cognition
that may carry evidence links; an observation is source material you cite, not a
fact; an inquiry is an open question you may answer later. Objects have one
of six kinds: person (a natural person), organization (a group with members
and decisions), place (a location on the map), event (a fact at a known
time), concept (an abstract meaning without a time), and entity (the
fallback for durable things that fit none of the above). Choose the kind
that names what the referent is; entity is the fallback, not the default. A
type_key is a domain refinement (team, player, match, tournament, season,
version) that may combine with any kind — never an identity key; prefer
domain-neutral words over prefixed ones.
Events are time anchors. {EVENT_REIFICATION_RULE} Carry an event's time in
event_time_start / event_time_end. Momentary statements with no durable
meaning stay assertions with event_time, not event nodes. Events declare
their participants explicitly as edges to objects you created or referenced;
the host never infers participants from a title or a literal.
Qualifiers are normalized bounded keys and values; assertion qualifiers
support only the keys role, language, community, scope, and granularity.
Notes without a qualifier slot (for example, a source conflict) go into the
assertion literal, never into a new qualifier key.
{IDENTITY_MODEL_SENTENCE}
Search hits are candidates, never auto-merge authorization: they tell you what
exists and why it matched, and you decide identity. Candidates never rewrite
your proposal.
On conflict or repair, resubmit with corrected references, add, remove, or
demote an identity alias (demotion records the name as name usage), omit the
conflicting item with its dependencies, or leave the question open as an
inquiry. An honest empty delta is better than fabricated cognition."""


def render_contract_text() -> str:
    """Render the short, domain-neutral graph cognition contract text.

    Covers the roles of objects/assertions/observations/inquiries, the six
    kinds plus ``type_key``, events as time anchors with the build/stay-
    assertion boundary, canonical/identity alias/name usage, search hits as
    candidates rather than auto-merge authorization, explicit event
    participants, the allowed conflict/repair actions, and the honest-empty-
    delta rule. Uses the same terms as the ``graph_contract`` normalizers
    without duplicating their rule bodies.
    """
    return _CONTRACT_BODY


_EXAMPLE_GROUPS: tuple[tuple[str, str, str], ...] = (
    (
        "Bounded multi-party occurrence",
        "Wrong: a preserved occurrence is compressed into a direct edge "
        "between two participants, with its date and result hidden in a role "
        "qualifier.",
        "Right: create or reference each recurring participant, create the "
        "bounded event, connect the event to every known participant, and put "
        "the result and other occurrence-specific details on the event. This "
        "same shape applies to an acquisition, an appointment, an exchange, or "
        "any other durable multi-party occurrence.",
    ),
    (
        "Same-name places",
        "Wrong: two places share one name, and one search hit merges them into "
        "a single object.",
        "Right: both objects stay separate, search returns both as candidates, "
        "and you disambiguate by kind, relations, time, and evidence.",
    ),
    (
        "Academic term alias",
        "Wrong: a nickname used in one community becomes the concept's identity "
        "alias.",
        "Right: the stable academic term is the identity alias; the community "
        "use is a name_usage assertion with community and language qualifiers.",
    ),
    (
        "Regulation versions",
        "Wrong: two versions of a regulation collapse into one object because "
        "they share a name and a date.",
        "Right: each version is its own concept with a type_key and "
        "granularity qualifiers; the versions coexist and a later version "
        "supersedes an earlier one explicitly.",
    ),
    (
        "Community nickname",
        "Wrong: a fan nickname becomes a world-unique alias.",
        "Right: the nickname is a name_usage assertion on the entity with "
        "community, period, and language qualifiers — never an identity key.",
    ),
    (
        "Single-entity event",
        "Wrong: a retirement, a milestone, or a release survives only as an "
        "assertion with a time, so nothing anchors the moment later wakes "
        "refer to.",
        "Right: the moment becomes an event object with event_time; the "
        "entity anchors through an assertion (subject = event, object = the "
        "entity). The shape is domain-neutral: a player's retirement is a "
        "retirement event with the player anchored to it.",
    ),
)


def render_contract_examples() -> str:
    """Render the six domain-neutral write-shape example groups.

    Each group pairs one counter-example with one positive graph structure:
    bounded multi-party occurrence (explicit event participants), same-name places (alias
    ambiguity resolved through candidates), academic term alias (stable term
    vs. community name usage), regulation versions (revision kept separate),
    community nickname (never an identity key), and single-entity event (a
    state-changing moment anchored as an event object).
    """
    blocks = []
    for index, (title, wrong, right) in enumerate(_EXAMPLE_GROUPS, start=1):
        blocks.append(f"{index}. {title}.\n{wrong}\n{right}")
    return "\n\n".join(blocks)
