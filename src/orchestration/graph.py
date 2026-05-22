from __future__ import annotations

from typing import Any, TypedDict

from src.orchestration.nodes import DebateNodeRunner, coerce_state, dump_state, should_continue
from src.orchestration.state import DebateState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - direct fallback remains usable for diagnostics.
    END = "END"  # type: ignore
    START = "START"  # type: ignore
    StateGraph = None  # type: ignore


class DebateGraphState(TypedDict, total=False):
    run_id: str
    created_at: str
    user_prompt: str
    mode: str
    max_iterations: int
    active_models: list[str]
    current_iteration: int
    initial_answers: dict[str, str]
    critiques: dict[str, list[str]]
    refinements: dict[str, list[str]]
    turns: list[dict[str, Any]]
    final_answer: str | None
    errors: list[str]


class DebateGraph:
    def __init__(self, settings: dict, progress_callback=None) -> None:
        self.runner = DebateNodeRunner(settings, progress_callback=progress_callback)
        self.compiled_graph = self._compile_graph()

    def _wrap(self, method):
        async def node(state: DebateGraphState) -> DebateGraphState:
            result = await method(coerce_state(state))
            return dump_state(result)

        return node

    def _compile_graph(self):
        if StateGraph is None:
            return None

        builder = StateGraph(DebateGraphState)
        builder.add_node("initialize_browser_sessions", self._wrap(self.runner.initialize_browser_sessions))
        builder.add_node("initial_answer_round", self._wrap(self.runner.initial_answer_round))
        builder.add_node("critique_round", self._wrap(self.runner.critique_round))
        builder.add_node("refinement_round", self._wrap(self.runner.refinement_round))
        builder.add_node("final_synthesis", self._wrap(self.runner.final_synthesis))
        builder.add_node("log_result", self._wrap(self.runner.log_result))

        builder.add_edge(START, "initialize_browser_sessions")
        builder.add_edge("initialize_browser_sessions", "initial_answer_round")
        builder.add_edge("initial_answer_round", "critique_round")
        builder.add_edge("critique_round", "refinement_round")
        builder.add_conditional_edges(
            "refinement_round",
            should_continue,
            {
                "continue": "critique_round",
                "finish": "final_synthesis",
            },
        )
        builder.add_edge("final_synthesis", "log_result")
        builder.add_edge("log_result", END)
        return builder.compile()

    async def ainvoke(self, initial_state: DebateState) -> DebateState:
        if self.compiled_graph is not None:
            result = await self.compiled_graph.ainvoke(dump_state(initial_state))
            return coerce_state(result)

        state = await self.runner.initialize_browser_sessions(initial_state)
        state = await self.runner.initial_answer_round(state)
        state = await self.runner.critique_round(state)
        state = await self.runner.refinement_round(state)
        while should_continue(state) == "continue":
            state = await self.runner.critique_round(state)
            state = await self.runner.refinement_round(state)
        state = await self.runner.final_synthesis(state)
        state = await self.runner.log_result(state)
        return state
