"""LangGraph task planner: Qwen3-14B plans a mission by calling tools over the Hydra scene graph.

The model gets the task instruction and five tools (paper Sec. III):

    get_planning_context   task instruction, robot, rooms, what kinds of objects the kitchen contains
    find_entities          objects whose label matches a query, nearest first, positions relative to the robot
    get_room_route         rooms the base passes through to reach an object (object -> place -> room links)
    query_safety_context   one object's box, its neighbours and the free space around it
    commit_plan            submit the plan; it is checked and errors come back as the tool result

The loop (agent -> tools -> agent ...) ends at the first accepted commit or after MAX_TURNS model calls.
The tools read only the scene graph and the task instruction: ids are Hydra's (O(12)), names are label names.

The model runs on the local vLLM server (scripts/llm/serve_qwen.sh) with native tool calling. The committed plan
is returned in the plans/*.json format with roles {"node": "O(12)"}, so run_episode can execute it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Literal

import numpy as np
import spark_dsg as dsg
from langchain_core.messages import HumanMessage, SystemMessage, messages_to_dict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from sesame_lite.scene_graph import SceneGraph

QWEN_URL = os.environ.get("SESAME_QWEN_URL", "http://127.0.0.1:8001/v1")
MAX_TURNS = 16
FIND_LIMIT = 15  # entities per find_entities answer
NEARBY_M = 1.0  # query_safety_context lists objects within this gap
SKILLS = ("navigate", "open", "close", "pick", "place", "pick_and_place", "turn_on", "turn_off")

# Qwen3 model card, thinking mode: temperature 0.6, top_p 0.95, top_k 20. Seed fixed for repeatability.
MODEL_SETTINGS = dict(model="qwen3-14b", temperature=0.6, top_p=0.95, seed=0, max_tokens=6000,
                      extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})

SYSTEM_PROMPT = (
    "You are the task planner of a mobile manipulator in a kitchen: a wheeled base that can drive in any "
    "direction and one arm that reaches about 1 m from the base. You see the kitchen only through tools that "
    "query its 3D scene graph. Positions are relative to the robot: ahead_m is in front of it, left_m to its "
    "left, height_m above the floor.\n\n"
    "A plan is an ordered list of segments. A NAVIGATION segment drives the base to its target; the arm does "
    "not move. A VLA segment keeps the base where it is while a vision-language-action policy performs a "
    "manipulation skill on its target (open, close, pick, place, pick_and_place, turn_on, turn_off) following "
    "the task instruction. Add a NAVIGATION segment before a VLA segment only if the target is out of the arm's "
    "reach from where the base will be. Roles name the entities the plan uses; each segment refers to its "
    "target by role.\n\n"
    "A safety filter keeps the robot away from every entity in the scene graph, except the entities you exempt. "
    "For each segment, exempt only the entities the robot must touch to do the task (for example the drawer it "
    "must open). Everything you do not exempt stays protected, so exempt as little as possible. Use entity ids "
    "exactly as the tools return them. Call commit_plan once the plan is complete."
)


# --------------------------------------------------------------------------- what the tools read


class SceneQueries:
    """Answers the tools give, computed from the scene graph and the task instruction only."""

    def __init__(self, graph: SceneGraph, task_instruction: str):
        self.graph, self.task = graph, task_instruction
        self.committed: dict | None = None  # the accepted commit_plan arguments

    def relative(self, p) -> dict:
        """A world point in the robot's frame."""
        d = np.asarray(p[:2], dtype=float) - self.graph.robot_xy
        c, s = math.cos(self.graph.robot_yaw), math.sin(self.graph.robot_yaw)
        return {"ahead_m": round(c * d[0] + s * d[1], 2), "left_m": round(-s * d[0] + c * d[1], 2), "height_m": round(float(p[2]), 2)}

    def describe(self, node_id: str) -> dict:
        node = self.graph.nodes[node_id]
        lo, hi = node.aabb()
        gap_xy = np.maximum(np.maximum(lo[:2] - self.graph.robot_xy, self.graph.robot_xy - hi[:2]), 0.0)
        return {"entity_id": node_id, "label": node.category, **self.relative(node.center),
                "distance_m": round(float(np.linalg.norm(gap_xy)), 2), "size_m": [round(v, 2) for v in node.extent]}

    def check_id(self, entity_id: str) -> str | None:
        return None if entity_id in self.graph.nodes else f"unknown entity_id {entity_id!r}; use ids returned by find_entities"

    def room_of(self, node_id: str) -> str:
        """Hydra's hierarchy: object -> place (parent) -> room (parent)."""
        place = self.graph.dsg.get_node(self.graph.dsg_node(node_id).get_parent())
        return str(dsg.NodeSymbol(place.get_parent()))


class Role(BaseModel):
    role: str = Field(description="a short name for an entity the plan uses, e.g. 'target'")
    entity_id: str = Field(description="the entity id, e.g. 'O(12)'")


class PlanSegment(BaseModel):
    kind: Literal["NAVIGATION", "VLA"]
    skill: Literal[SKILLS]
    target_role: str = Field(description="the role of this segment's target")
    exempt: list[str] = Field(default_factory=list, description="entity ids the robot may touch in this segment")


def make_tools(q: SceneQueries) -> list:
    @tool
    def get_planning_context() -> str:
        """The task instruction, the robot, the rooms and how many entities of each label the kitchen has."""
        counts: dict[str, int] = {}
        for node in q.graph.nodes.values():
            counts[node.category] = counts.get(node.category, 0) + 1
        return json.dumps({"task_instruction": q.task, "rooms": q.graph.rooms, "robot_room": q.graph.rooms[0],
                           "arm_reach_m": 1.0, "entity_counts": dict(sorted(counts.items()))})

    @tool
    def find_entities(query: str) -> str:
        """Entities whose label contains a word of the query (e.g. 'drawer', 'fridge'), nearest first."""
        words = [w for w in query.lower().replace("_", " ").split() if len(w) >= 3]
        hits = [q.describe(nid) for nid, n in q.graph.nodes.items()
                if not words or any(w in n.category or w.rstrip("s") in n.category for w in words)]
        hits.sort(key=lambda h: (h["distance_m"], h["entity_id"]))
        return json.dumps({"query": query, "n_matches": len(hits), "entities": hits[:FIND_LIMIT]})

    @tool
    def get_room_route(entity_id: str) -> str:
        """The rooms the base passes through to reach an entity."""
        if (err := q.check_id(entity_id)) is not None:
            return json.dumps({"error": err})
        rooms = [q.graph.rooms[0], q.room_of(entity_id)]
        return json.dumps({"entity_id": entity_id, "route": list(dict.fromkeys(rooms)), "distance_m": q.describe(entity_id)["distance_m"]})

    @tool
    def query_safety_context(entity_id: str) -> str:
        """One entity's box, the entities within 1 m of it and the free space next to it."""
        if (err := q.check_id(entity_id)) is not None:
            return json.dumps({"error": err})
        lo, hi = q.graph.nodes[entity_id].aabb()
        nearby = []
        for nid, n in q.graph.nodes.items():
            if nid == entity_id:
                continue
            nlo, nhi = n.aabb()
            gap = float(np.linalg.norm(np.maximum(np.maximum(lo - nhi, nlo - hi), 0.0)))
            if gap <= NEARBY_M:
                nearby.append({"entity_id": nid, "label": n.category, "gap_m": round(gap, 2)})
        nearby.sort(key=lambda e: (e["gap_m"], e["entity_id"]))
        clearance = float(q.graph.dsg.get_node(q.graph.dsg_node(entity_id).get_parent()).attributes.distance)
        return json.dumps({**q.describe(entity_id), "nearby": nearby[:20], "free_space_clearance_m": round(clearance, 2)})

    @tool
    def commit_plan(roles: list[Role], segments: list[PlanSegment]) -> str:
        """Submit the final plan: roles (role name -> entity id) and the ordered segments."""
        errors, role_ids = [], {}
        for r in roles:
            if (err := q.check_id(r.entity_id)) is not None:
                errors.append(f"role {r.role!r}: {err}")
            role_ids[r.role] = r.entity_id
        if not segments:
            errors.append("a plan needs at least one segment")
        for i, s in enumerate(segments):
            if s.target_role not in role_ids:
                errors.append(f"segment {i}: target_role {s.target_role!r} is not one of the roles")
            if (s.kind == "NAVIGATION") != (s.skill == "navigate"):
                errors.append(f"segment {i}: skill 'navigate' goes with kind NAVIGATION, and only with it")
            errors += [f"segment {i}: exempt {err}" for e in s.exempt if (err := q.check_id(e)) is not None]
        if errors:
            return json.dumps({"accepted": False, "errors": errors})
        q.committed = {"roles": [r.model_dump() for r in roles], "segments": [s.model_dump() for s in segments]}
        return json.dumps({"accepted": True})

    return [get_planning_context, find_entities, get_room_route, query_safety_context, commit_plan]


def to_plan(committed: dict) -> dict:
    """Accepted commit_plan arguments -> the plans/*.json format (roles are scene-graph nodes)."""
    roles = {r["role"]: {"node": r["entity_id"]} for r in committed["roles"]}
    segments = []
    for s in committed["segments"]:
        exempt = [{"node": e} for e in s["exempt"]]
        if s["kind"] == "NAVIGATION":
            segments.append({"kind": "NAVIGATION", "skill": "navigate", "goal": {"role": s["target_role"], "standoff": True},
                             "exempt": exempt, "done": {"check": "base_near_goal", "radius": 0.2}})
        else:
            segments.append({"kind": "VLA", "skill": s["skill"], "targets": [s["target_role"]], "exempt": exempt, "done": {"check": "task_success"}})
    return {"roles": roles, "segments": segments}


# --------------------------------------------------------------------------- the LangGraph loop


class PlannerState(MessagesState):
    turns: int


def run_planner(graph: SceneGraph, task_instruction: str, *, max_turns: int = MAX_TURNS) -> dict:
    """Plan one mission. Returns the plan (or None), the full message transcript and the number of model turns."""
    queries = SceneQueries(graph, task_instruction)
    tools = make_tools(queries)
    # tool_choice="required": vLLM constrains every reply to a schema-valid tool call (unconstrained, Qwen3-14B's
    # calls with nested arguments came out as malformed JSON on the design seed)
    llm = ChatOpenAI(base_url=QWEN_URL, api_key="local", **MODEL_SETTINGS).bind_tools(tools, tool_choice="required")

    def agent(state: PlannerState) -> dict:
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply], "turns": state["turns"] + 1}

    def nudge(state: PlannerState) -> dict:  # the model answered in text: ask for a tool call
        return {"messages": [HumanMessage("Continue by calling one of the tools.")]}

    def after_agent(state: PlannerState) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END if state["turns"] >= max_turns else "nudge"

    def after_tools(state: PlannerState) -> str:
        return END if queries.committed is not None or state["turns"] >= max_turns else "agent"

    builder = StateGraph(PlannerState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("nudge", nudge)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, ["tools", "nudge", END])
    builder.add_conditional_edges("tools", after_tools, ["agent", END])
    builder.add_edge("nudge", "agent")
    app = builder.compile()

    first = HumanMessage(f"Task instruction: {task_instruction}\nPlan this task.")
    final = app.invoke({"messages": [first], "turns": 0}, {"recursion_limit": 4 * max_turns + 8})
    return {"plan": to_plan(queries.committed) if queries.committed else None,
            "committed": queries.committed,
            "turns": final["turns"],
            "failure": None if queries.committed else "no accepted commit_plan within the turn budget",
            "settings": {**MODEL_SETTINGS, "weights": "Qwen/Qwen3-14B-AWQ", "max_turns": max_turns},
            "system_prompt": SYSTEM_PROMPT,
            "transcript": messages_to_dict([SystemMessage(SYSTEM_PROMPT), *final["messages"]])}
