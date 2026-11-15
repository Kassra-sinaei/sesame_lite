"""Stage-2 semantic pruning classifier (port of signal_3d ``tool_calling/pruning_classifier.py``).

The contract is unchanged from the paper: one bounded structured call per
segment, a fixed three-label vocabulary, labels may only *remove* objects from
the geometric candidate set (see ``swept_volume.admitted_after_labels``), and
any decode failure fails closed to "obstacle" for every candidate.

What changed: the loopback vLLM/OpenAI-compatible server is replaced by Claude
Code's headless mode (``claude -p --json-schema``), which uses the laptop's
existing claude.ai login. No API key is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Mapping, Protocol, Sequence

PRUNING_CLASSIFIER_VERSION = "sesame-lite-pruning-classifier/1"
CLASSIFIER_LABELS = ("obstacle", "passable", "irrelevant")
MAXIMUM_CANDIDATES = 100

# Deterministic category priors for the offline classifier (same as signal_3d).
PASSABLE_CATEGORY_PRIORS = ("curtain", "curtains")

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class PruningCandidate:
    """One geometric candidate offered to the classifier."""

    stable_id: str
    name: str
    kind: str = "object"

    def __post_init__(self) -> None:
        for value, label in (
            (self.stable_id, "stable id"),
            (self.name, "name"),
            (self.kind, "kind"),
        ):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError(f"Pruning candidate {label} must be bounded non-empty text.")


@dataclass(frozen=True)
class PruningClassification:
    """Snapshot-bound labels for exactly the offered candidates."""

    labels: Mapping[str, str]
    parent_graph_snapshot_id: str
    classifier_version: str
    model_identity: str | None = None
    fallback_used: bool = False
    request_digest: str | None = None
    response_digest: str | None = None
    detail: str | None = None
    latency_s: float | None = None

    def __post_init__(self) -> None:
        if not self.parent_graph_snapshot_id:
            raise ValueError("Classification requires its snapshot binding.")
        for key, value in self.labels.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Classification keys must be stable IDs.")
            if value not in CLASSIFIER_LABELS:
                raise ValueError("Classification label vocabulary is fixed.")

    def as_dict(self) -> dict[str, object]:
        return {
            "classifier_version": self.classifier_version,
            "parent_graph_snapshot_id": self.parent_graph_snapshot_id,
            "labels": dict(self.labels),
            "model_identity": self.model_identity,
            "fallback_used": self.fallback_used,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "detail": self.detail,
            "latency_s": self.latency_s,
        }


class PruningCandidateClassifier(Protocol):
    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification: ...


def classifier_output_schema() -> dict[str, Any]:
    """Bounded JSON schema for the single structured reply (was the tool's parameters)."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAXIMUM_CANDIDATES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["stable_id", "label"],
                    "properties": {
                        "stable_id": {"type": "string", "minLength": 1, "maxLength": 256},
                        "label": {"type": "string", "enum": list(CLASSIFIER_LABELS)},
                    },
                },
            }
        },
    }


# Verbatim from signal_3d except that the tool call became a structured reply.
CLASSIFIER_SYSTEM_PROMPT = (
    "You label objects for a mobile manipulator's collision-constraint "
    "pruning. You receive the task goal and a list of scene objects "
    "(stable_id, semantic name, kind); names may include measured "
    "dimensions. Reply with exactly one label per object. "
    "Use 'passable' only for objects the wheeled robot can physically "
    "traverse without damage: hanging soft material it can push through "
    "(a curtain, a draped sheet) or flat floor coverings only a few "
    "centimeters tall it can drive over (a mat, a rug, a doormat). Rigid "
    "objects, containers, and furniture are never passable, and neither "
    "are loose items that would be displaced by contact (a box, a "
    "pillow, a rolled-up mat). Use 'irrelevant' only for objects that "
    "cannot interact with this task. When in doubt, use 'obstacle'. "
    "Never omit or invent an object. 'obstacle' keeps its collision "
    "constraints, 'passable' marks an object the robot may physically pass "
    "through, 'irrelevant' marks an object that cannot interact with this "
    "task's route."
)


def _validated_candidates(candidates: Sequence[PruningCandidate]) -> tuple[PruningCandidate, ...]:
    items = tuple(candidates)
    if not items or len(items) > MAXIMUM_CANDIDATES:
        raise ValueError(f"Classifier accepts 1..{MAXIMUM_CANDIDATES} candidates.")
    ids = [item.stable_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate stable IDs must be unique.")
    return items


def _validated_all(candidates: Sequence[PruningCandidate]) -> tuple[PruningCandidate, ...]:
    """Any number of candidates, checked in chunks of MAXIMUM_CANDIDATES (for classifiers that ask in chunks)."""
    items = tuple(candidates)
    for k in range(0, max(len(items), 1), MAXIMUM_CANDIDATES):
        _validated_candidates(items[k:k + MAXIMUM_CANDIDATES])
    if len({item.stable_id for item in items}) != len(items):
        raise ValueError("Candidate stable IDs must be unique.")
    return items


def _validated_goal(task_goal: str) -> str:
    if not isinstance(task_goal, str) or not task_goal.strip() or len(task_goal) > 2000:
        raise ValueError("Task goal must be bounded non-empty text.")
    return task_goal.strip()


def request_payload(candidates: Sequence[PruningCandidate], task_goal: str) -> str:
    """Sorted-key JSON user message, identical layout to signal_3d."""

    return json.dumps(
        {
            "task_goal": task_goal,
            "objects": [
                {"stable_id": c.stable_id, "name": c.name, "kind": c.kind} for c in candidates
            ],
        },
        sort_keys=True,
    )


class CategoryPriorClassifier:
    """Deterministic category-map classifier (offline and fail-safe path)."""

    passable_categories: tuple[str, ...] = PASSABLE_CATEGORY_PRIORS
    irrelevant_categories: tuple[str, ...] = ()
    classifier_version: str = f"{PRUNING_CLASSIFIER_VERSION}+category-prior"

    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification:
        _validated_goal(task_goal)
        items = tuple(candidates)
        # MAXIMUM_CANDIDATES is the LLM classifiers' reply-size limit; this rule labels every object on its own, so a long
        # path's corridor with more candidates is checked in chunks (same labels)
        for k in range(0, max(len(items), 1), MAXIMUM_CANDIDATES):
            _validated_candidates(items[k:k + MAXIMUM_CANDIDATES])
        if len({item.stable_id for item in items}) != len(items):
            raise ValueError("Candidate stable IDs must be unique.")
        passable = {v.strip().lower() for v in self.passable_categories}
        irrelevant = {v.strip().lower() for v in self.irrelevant_categories}
        labels = {}
        for item in items:
            category = item.name.split(" (")[0].strip().lower()
            if category in passable:
                labels[item.stable_id] = "passable"
            elif category in irrelevant:
                labels[item.stable_id] = "irrelevant"
            else:
                labels[item.stable_id] = "obstacle"
        return PruningClassification(
            labels=labels,
            parent_graph_snapshot_id=parent_graph_snapshot_id,
            classifier_version=self.classifier_version,
        )


class ClassifierDecodeError(RuntimeError):
    """The reply was not a valid labelling of exactly the offered candidates."""


def decode_classifier_labels(payload: Any, offered_ids: Sequence[str]) -> dict[str, str]:
    """Exact-cover decode: fixed vocabulary, every offered ID once, nothing else."""

    if not isinstance(payload, Mapping) or set(payload) != {"labels"}:
        raise ClassifierDecodeError("Reply must be an object with the single key 'labels'.")
    rows = payload["labels"]
    if not isinstance(rows, list) or not rows:
        raise ClassifierDecodeError("'labels' must be a non-empty list.")
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"stable_id", "label"}:
            raise ClassifierDecodeError("Each row needs exactly stable_id and label.")
        sid, lab = row["stable_id"], row["label"]
        if lab not in CLASSIFIER_LABELS:
            raise ClassifierDecodeError(f"Label {lab!r} outside the vocabulary.")
        if sid in labels:
            raise ClassifierDecodeError(f"Duplicate stable_id {sid!r}.")
        labels[sid] = lab
    if set(labels) != set(offered_ids):
        raise ClassifierDecodeError("Reply must cover exactly the offered stable IDs.")
    return labels


@dataclass
class ClaudeCodeClassifier:
    """One structured ``claude -p`` call per segment.

    ``--bare`` must not be used: it drops the claude.ai login. ``--tools ""``
    removes every tool so the reply is text validated against the schema.
    """

    model: str = DEFAULT_CLAUDE_MODEL
    effort: str = "low"
    timeout_s: float = 120.0
    claude_binary: str = "claude"
    classifier_version: str = f"{PRUNING_CLASSIFIER_VERSION}+claude-code"
    extra_args: tuple[str, ...] = ()
    last_raw_result: dict[str, Any] | None = field(default=None, repr=False)

    def _run(self, user_message: str) -> dict[str, Any]:
        cmd = [
            self.claude_binary,
            "-p",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--tools",
            "",
            "--system-prompt",
            CLASSIFIER_SYSTEM_PROMPT,
            "--json-schema",
            json.dumps(classifier_output_schema()),
            *self.extra_args,
            user_message,
        ]
        # Claude Code refuses to nest inside another Claude Code session unless
        # these variables are cleared; they are harmless to drop.
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout_s, env=env, check=False
        )
        if completed.returncode != 0 and not completed.stdout.strip():
            raise ClassifierDecodeError(
                f"claude exited {completed.returncode}: {completed.stderr.strip()[:400]}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ClassifierDecodeError(f"claude output is not JSON: {exc}") from exc
        self.last_raw_result = result
        if result.get("is_error"):
            raise ClassifierDecodeError(f"claude error: {str(result.get('result'))[:400]}")
        return result

    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification:
        goal = _validated_goal(task_goal)
        items = _validated_candidates(candidates)
        user_message = request_payload(items, goal)
        t0 = time.perf_counter()
        result = self._run(user_message)
        latency = time.perf_counter() - t0
        structured = result.get("structured_output")
        if structured is None:
            # Older CLI builds put the JSON text in "result".
            try:
                structured = json.loads(str(result.get("result", "")))
            except json.JSONDecodeError as exc:
                raise ClassifierDecodeError("No structured_output in claude reply.") from exc
        labels = decode_classifier_labels(structured, [c.stable_id for c in items])
        return PruningClassification(
            labels=labels,
            parent_graph_snapshot_id=parent_graph_snapshot_id,
            classifier_version=self.classifier_version,
            model_identity=self.model,
            request_digest=hashlib.sha256(user_message.encode()).hexdigest(),
            response_digest=hashlib.sha256(
                json.dumps(structured, sort_keys=True).encode()
            ).hexdigest(),
            latency_s=latency,
        )


@dataclass
class ConservativeFallbackClassifier:
    """Retry once, then label everything 'obstacle' (fail closed)."""

    inner: PruningCandidateClassifier
    retries: int = 1
    classifier_version: str = f"{PRUNING_CLASSIFIER_VERSION}+conservative-fallback"
    failures: list[str] = field(default_factory=list)

    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification:
        items = _validated_all(candidates)
        last_error = None
        for _ in range(self.retries + 1):
            try:
                return self.inner.classify(
                    candidates=items,
                    task_goal=task_goal,
                    parent_graph_snapshot_id=parent_graph_snapshot_id,
                )
            except (ClassifierDecodeError, subprocess.TimeoutExpired, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.failures.append(last_error)
        return PruningClassification(
            labels={item.stable_id: "obstacle" for item in items},
            parent_graph_snapshot_id=parent_graph_snapshot_id,
            classifier_version=self.classifier_version,
            fallback_used=True,
            detail=last_error,
        )


@dataclass
class StickyPassabilityClassifier:
    """Remember passable/irrelevant verdicts for the rest of the episode."""

    inner: PruningCandidateClassifier
    classifier_version: str = f"{PRUNING_CLASSIFIER_VERSION}+sticky"
    _memory: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self._memory.clear()

    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification:
        items = _validated_all(candidates)
        remembered = {c.stable_id: self._memory[c.stable_id] for c in items if c.stable_id in self._memory}
        fresh = [c for c in items if c.stable_id not in remembered]
        if fresh:
            result = self.inner.classify(
                candidates=fresh, task_goal=task_goal, parent_graph_snapshot_id=parent_graph_snapshot_id
            )
            labels = dict(result.labels)
            for sid, lab in labels.items():
                if lab != "obstacle":
                    self._memory[sid] = lab
        else:
            result = PruningClassification(
                labels={},
                parent_graph_snapshot_id=parent_graph_snapshot_id,
                classifier_version=self.classifier_version,
            )
            labels = {}
        labels.update(remembered)
        return replace(result, labels=labels, classifier_version=f"{result.classifier_version}+sticky")



@dataclass
class QwenClassifier:
    """The same single structured call, answered by Qwen3-14B on the local vLLM server (scripts/llm/serve_qwen.sh).

    Same prompt, schema, payload and exact-cover decode as the Claude path. Thinking off, with the Qwen3 model
    card's sampling for that mode (temperature 0.7, top_p 0.8, top_k 20); seed fixed for repeatability.
    """

    base_url: str = os.environ.get("SESAME_QWEN_URL", "http://127.0.0.1:8001/v1")
    model: str = "qwen3-14b"
    timeout_s: float = 120.0
    classifier_version: str = f"{PRUNING_CLASSIFIER_VERSION}+qwen3-14b"

    def classify(
        self,
        *,
        candidates: Sequence[PruningCandidate],
        task_goal: str,
        parent_graph_snapshot_id: str,
    ) -> PruningClassification:
        # One plain HTTP request with the standard library (no client package needed in the episode process).
        import urllib.error
        import urllib.request

        goal = _validated_goal(task_goal)
        everything = tuple(candidates)
        if not everything:
            raise ValueError("Classifier accepts 1..100 candidates.")
        # the reply schema holds at most MAXIMUM_CANDIDATES labels: a longer list (a long path's corridor) is asked in
        # chunks of that size, one request each with the same prompt, and the labels are merged
        labels, digests, t0 = {}, [], time.perf_counter()
        for start in range(0, len(everything), MAXIMUM_CANDIDATES):
            items = _validated_candidates(everything[start:start + MAXIMUM_CANDIDATES])
            user_message = request_payload(items, goal)
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": classifier_output_schema(), "strict": True}},
                "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 0, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    reply = json.loads(response.read())
                structured = json.loads(reply["choices"][0]["message"]["content"] or "")
            except (OSError, KeyError, json.JSONDecodeError) as exc:  # URLError and timeouts are OSErrors
                raise ClassifierDecodeError(f"qwen: {exc}") from exc
            labels.update(decode_classifier_labels(structured, [c.stable_id for c in items]))
            digests.append((hashlib.sha256(user_message.encode()).hexdigest(), hashlib.sha256(json.dumps(structured, sort_keys=True).encode()).hexdigest()))
        latency = time.perf_counter() - t0
        return PruningClassification(
            labels=labels,
            parent_graph_snapshot_id=parent_graph_snapshot_id,
            classifier_version=self.classifier_version,
            model_identity="Qwen/Qwen3-14B-AWQ",
            request_digest=hashlib.sha256("".join(d[0] for d in digests).encode()).hexdigest(),
            response_digest=hashlib.sha256("".join(d[1] for d in digests).encode()).hexdigest(),
            latency_s=latency,
        )


# ---------------------------------------------------------------------- pair-level pruning (round 5 of the evaluation)

ROBOT_PARTS = ("base", "arm", "hand", "fingers")

# Version 2, written once on 2026-11-16 and frozen before any model run with it (SHA-256 in tuning_amendments.md). Version
# 1 (2026-11-13) gave no positions, so the model could not tell which of several drawers or cabinets the task meant; the
# only change is the sentence on where each object is.
PAIR_SYSTEM_PROMPT = (
    "You choose collision pairs for a mobile manipulator's safety filter. The filter keeps every robot part away from "
    "every scene object. The robot parts are: base (the wheeled base), arm (the arm links), hand (the hand), fingers "
    "(the gripper's two fingers). You receive the task goal of one manipulation step and a list of scene objects "
    "(stable_id, semantic name, kind); names may include measured dimensions. The robot stands where it will do the "
    "task and faces it; every object comes with its distance from the robot, its direction as seen from the robot and "
    "the height of its centre, and the list is ordered from the nearest object to the farthest. "
    "Reply with the (part, object) pairs whose "
    "separation constraint must be removed so that the task can be done, because the task needs that part to touch the "
    "object or to reach into it. Every pair you do not list stays active. List only pairs the task requires; when in "
    "doubt, leave a pair active. Use only the given stable_ids and parts."
)


def pair_output_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["drop"],
        "properties": {"drop": {"type": "array", "maxItems": 4 * MAXIMUM_CANDIDATES, "items": {
            "type": "object", "additionalProperties": False, "required": ["part", "stable_id"],
            "properties": {"part": {"type": "string", "enum": list(ROBOT_PARTS)}, "stable_id": {"type": "string", "minLength": 1, "maxLength": 256}}}}},
    }


@dataclass(frozen=True)
class PairClassification:
    pairs: tuple[tuple[str, str], ...]  # (robot part, stable_id) to drop; empty when the reply could not be used
    latency_s: float
    raw_reply: str
    error: str | None  # why the reply could not be used (then no pair is dropped)


class QwenPairClassifier:
    """Qwen3-14B on the local vLLM server (as QwenClassifier: same endpoint, sampling and seed) chooses the (robot part,
    object) pairs to drop for one manipulation step, from the task goal, the object list and the four robot parts."""

    base_url: str = QwenClassifier.base_url
    model: str = QwenClassifier.model
    timeout_s: float = 120.0

    def classify_pairs(self, *, candidates: Sequence[PruningCandidate], task_goal: str, where: Mapping[str, dict]) -> PairClassification:
        """``where``: stable_id -> {"distance_m", "direction", "height_m"} relative to the robot, from the map."""
        import urllib.request

        goal = _validated_goal(task_goal)
        items = tuple(sorted(candidates, key=lambda c: where[c.stable_id]["distance_m"]))
        payload = json.dumps({"task_goal": goal, "robot_parts": list(ROBOT_PARTS),
                              "objects": [{"stable_id": c.stable_id, "name": c.name, "kind": c.kind, **where[c.stable_id]} for c in items]})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": PAIR_SYSTEM_PROMPT}, {"role": "user", "content": payload}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "pairs", "schema": pair_output_schema(), "strict": True}},
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 0, "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0, raw, error, pairs = time.perf_counter(), "", None, ()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read())["choices"][0]["message"]["content"] or ""
            reply = json.loads(raw)
            ids = {c.stable_id for c in items}
            got = [(d["part"], d["stable_id"]) for d in reply["drop"]]
            bad = [p for p in got if p[0] not in ROBOT_PARTS or p[1] not in ids]
            if bad:
                error = f"unknown part or object in the reply: {bad[:5]}"
            else:
                pairs = tuple(dict.fromkeys(got))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return PairClassification(pairs=pairs, latency_s=time.perf_counter() - t0, raw_reply=raw, error=error)
