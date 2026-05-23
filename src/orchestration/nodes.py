from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
from typing import Any

from src.browser.base_chat_adapter import BaseChatAdapter
from src.browser.chatgpt_adapter import ChatGPTAdapter
from src.browser.claude_adapter import ClaudeAdapter
from src.browser.edge_manager import EdgeManager
from src.browser.gemini_adapter import GeminiAdapter
from src.orchestration.state import DebateState
from src.prompts.prompt_loader import PromptLoader
from src.storage.logger import DebateJSONLLogger
from src.storage.sqlite_store import SQLiteStore


AdapterClass = type[BaseChatAdapter]
ProgressCallback = Callable[..., Awaitable[None] | None]


ADAPTERS: dict[str, AdapterClass] = {
    "chatgpt": ChatGPTAdapter,
    "gemini": GeminiAdapter,
    "claude": ClaudeAdapter,
}


def coerce_state(state: DebateState | dict[str, Any]) -> DebateState:
    if isinstance(state, DebateState):
        return state
    if hasattr(DebateState, "model_validate"):
        return DebateState.model_validate(state)
    return DebateState.parse_obj(state)


def dump_state(state: DebateState) -> dict[str, Any]:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state.dict()


def should_continue(state: DebateState | dict[str, Any]) -> str:
    debate_state = coerce_state(state)
    if debate_state.mode == "normal":
        return "finish"
    if debate_state.final_answer:
        return "finish"
    if len(debate_state.active_models) < 2:
        return "finish"
    if debate_state.current_iteration < debate_state.max_iterations:
        return "continue"
    return "finish"


class DebateNodeRunner:
    def __init__(self, settings: dict, progress_callback: ProgressCallback | None = None) -> None:
        self.settings = settings
        self.sites = settings.get("model_sites", {})
        self.prompt_loader = PromptLoader()
        self.edge_manager = EdgeManager(settings)
        self.adapters: dict[str, BaseChatAdapter] = {}
        self.jsonl_logger = DebateJSONLLogger(settings)
        self.sqlite_store = SQLiteStore(settings)
        self.progress_callback = progress_callback

    async def initialize_browser_sessions(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        browser_mode = self.settings.get("browser", {}).get("mode", "existing_edge_cdp")
        if browser_mode == "existing_edge_cdp":
            await self._progress("Attaching to existing Edge AI tabs", debate_state)
        else:
            await self._progress("Opening Edge browser sessions", debate_state)

        working_models: list[str] = []
        for model_key in list(debate_state.active_models):
            try:
                await self._progress(f"{self._model_label(model_key)}: checking tab", debate_state)
                site = self.sites[model_key]
                page = None
                if browser_mode == "existing_edge_cdp":
                    context, page = await self.edge_manager.attach_to_existing_page(model_key, site["url"])
                else:
                    profile_path = self.settings["profiles"][model_key]
                    context = await self.edge_manager.launch_context(model_key, profile_path)
                adapter_class = ADAPTERS[model_key]
                adapter = adapter_class(model_key, site, context, self.settings, page=page)
                await adapter.open()
                self.adapters[model_key] = adapter
                working_models.append(model_key)
                await self._progress(f"{self._model_label(model_key)}: connected", debate_state)
            except Exception as exc:  # noqa: BLE001 - convert browser failures into state errors.
                debate_state.record_turn(
                    model=model_key,
                    iteration=0,
                    phase="initialize",
                    prompt="",
                    response="",
                    error=str(exc),
                )
                await self._progress(f"{self._model_label(model_key)}: connection failed", debate_state)

        debate_state.active_models = working_models
        min_models = 1 if debate_state.mode == "normal" else 2
        if len(debate_state.active_models) < min_models:
            debate_state.final_answer = (
                f"CouncilAI needs at least {min_models} working AI tab"
                f"{'' if min_models == 1 else 's'}. "
                "Open ChatGPT, Gemini, and Claude in the Edge instance started with "
                "--remote-debugging-port=9222, then retry."
            )
        return debate_state

    async def initial_answer_round(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        if debate_state.final_answer:
            return debate_state

        await self._progress("Collecting initial answers", debate_state)
        prompt_by_model: dict[str, str] = {}
        for model_key in debate_state.active_models:
            role = self.sites[model_key].get("role", "")
            prompt_by_model[model_key] = self.prompt_loader.render(
                "initial_answer",
                {
                    "USER_PROMPT": debate_state.user_prompt,
                    "MODEL_ROLE": role,
                },
            )

        response_pairs = await asyncio.gather(
            *[
                self._ask_model(debate_state, model_key, 0, "initial_answer", prompt)
                for model_key, prompt in prompt_by_model.items()
            ]
        )

        working_models: list[str] = []
        for model_key, response in zip(prompt_by_model.keys(), response_pairs, strict=False):
            if response:
                debate_state.initial_answers[model_key] = response
                working_models.append(model_key)

        debate_state.active_models = working_models
        min_models = 1 if debate_state.mode == "normal" else 2
        if len(debate_state.active_models) < min_models:
            debate_state.final_answer = (
                f"Fewer than {min_models} selected model"
                f"{'' if min_models == 1 else 's'} returned an initial answer. "
                "Check login/session status and selector errors, then retry."
            )
        return debate_state

    async def critique_round(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        if debate_state.final_answer or len(debate_state.active_models) < 2:
            return debate_state

        debate_state.current_iteration += 1
        await self._progress(f"Running critique round {debate_state.current_iteration}", debate_state)
        working_models: list[str] = []

        for model_key in list(debate_state.active_models):
            prompt = self.prompt_loader.render(
                "critique",
                {
                    "USER_PROMPT": debate_state.user_prompt,
                    "SELF_PREVIOUS_ANSWER": debate_state.latest_answer_for(model_key),
                    "OTHER_MODEL_ANSWERS": self._other_model_answers(debate_state, model_key),
                },
            )
            response = await self._ask_model(
                debate_state,
                model_key,
                debate_state.current_iteration,
                "critique",
                prompt,
            )
            if response:
                debate_state.critiques.setdefault(model_key, []).append(response)
                working_models.append(model_key)

        debate_state.active_models = working_models
        if len(debate_state.active_models) < 2:
            debate_state.final_answer = (
                "Fewer than two models completed the critique round. "
                "The debate cannot continue reliably."
            )
        return debate_state

    async def refinement_round(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        if debate_state.final_answer or len(debate_state.active_models) < 2:
            return debate_state

        await self._progress(f"Running refinement round {debate_state.current_iteration}", debate_state)
        working_models: list[str] = []

        for model_key in list(debate_state.active_models):
            prompt = self.prompt_loader.render(
                "refinement",
                {
                    "USER_PROMPT": debate_state.user_prompt,
                    "SELF_PREVIOUS_ANSWER": debate_state.latest_answer_for(model_key),
                    "ALL_CRITIQUES": self._all_critiques(debate_state),
                },
            )
            response = await self._ask_model(
                debate_state,
                model_key,
                debate_state.current_iteration,
                "refinement",
                prompt,
            )
            if response:
                debate_state.refinements.setdefault(model_key, []).append(response)
                working_models.append(model_key)

        debate_state.active_models = working_models
        if len(debate_state.active_models) < 2:
            debate_state.final_answer = (
                "Fewer than two models completed refinement. "
                "The debate cannot produce a reliable synthesis."
            )
        return debate_state

    async def final_synthesis(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        if debate_state.final_answer:
            return debate_state

        final_responses = self._final_model_responses(debate_state)
        if not final_responses.strip():
            debate_state.final_answer = "No model responses were available for final synthesis."
            return debate_state

        await self._progress("Generating final synthesis", debate_state)
        prompt = self.prompt_loader.render(
            "final_synthesis",
            {
                "USER_PROMPT": debate_state.user_prompt,
                "FINAL_MODEL_RESPONSES": final_responses,
                "DEBATE_HISTORY": self._debate_history(debate_state),
            },
        )

        for model_key in ("chatgpt", "claude", "gemini"):
            if model_key not in debate_state.active_models:
                continue
            response = await self._ask_model(
                debate_state,
                model_key,
                debate_state.current_iteration,
                "final_synthesis",
                prompt,
            )
            if response:
                debate_state.final_answer = response
                await self._progress("Final synthesis complete", debate_state)
                return debate_state

        debate_state.final_answer = (
            "Final synthesis failed in all available models. "
            "Check the saved logs for model-specific errors."
        )
        await self._progress("Final synthesis failed", debate_state)
        return debate_state

    async def log_result(self, state: DebateState | dict[str, Any]) -> DebateState:
        debate_state = coerce_state(state)
        await self._progress("Saving logs", debate_state)
        try:
            self.jsonl_logger.write(debate_state)
            self.sqlite_store.save(debate_state)
        finally:
            await self.edge_manager.stop()
        return debate_state

    async def _ask_model(
        self,
        state: DebateState,
        model_key: str,
        iteration: int,
        phase: str,
        prompt: str,
    ) -> str | None:
        adapter = self.adapters.get(model_key)
        if adapter is None:
            state.record_turn(model_key, iteration, phase, prompt, "", "adapter not initialized")
            await self._progress(f"{self._model_label(model_key)}: {phase.replace('_', ' ')} failed", state)
            return None

        phase_label = phase.replace("_", " ")
        await self._progress(f"{self._model_label(model_key)}: {phase_label} started", state)
        try:
            response = await adapter.ask(prompt)
            state.record_turn(model_key, iteration, phase, prompt, response)
            await self._progress(f"{self._model_label(model_key)}: {phase_label} complete", state)
            return response
        except Exception as exc:  # noqa: BLE001 - one model failure must not crash the app.
            state.record_turn(model_key, iteration, phase, prompt, "", str(exc))
            await self._progress(f"{self._model_label(model_key)}: {phase_label} failed", state)
            return None

    async def _progress(self, message: str, state: DebateState | None = None) -> None:
        if self.progress_callback is None:
            return
        callback = self.progress_callback
        try:
            signature = inspect.signature(callback)
            supports_state = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            ) or len(signature.parameters) >= 2
        except (TypeError, ValueError):
            supports_state = False

        result = callback(message, state) if supports_state else callback(message)
        if hasattr(result, "__await__"):
            await result

    def _model_label(self, model_key: str) -> str:
        return self.sites.get(model_key, {}).get("name", model_key)

    def _other_model_answers(self, state: DebateState, current_model: str) -> str:
        sections = []
        for model_key in state.active_models:
            if model_key == current_model:
                continue
            answer = state.latest_answer_for(model_key)
            if answer:
                sections.append(f"## {self._model_label(model_key)}\n{answer}")
        return "\n\n".join(sections)

    def _all_critiques(self, state: DebateState) -> str:
        sections = []
        for model_key, critiques in state.critiques.items():
            for index, critique in enumerate(critiques, start=1):
                sections.append(f"## {self._model_label(model_key)} critique {index}\n{critique}")
        return "\n\n".join(sections)

    def _final_model_responses(self, state: DebateState) -> str:
        sections = []
        for model_key in state.active_models:
            answer = state.latest_answer_for(model_key)
            if answer:
                sections.append(f"## {self._model_label(model_key)}\n{answer}")
        return "\n\n".join(sections)

    def _debate_history(self, state: DebateState) -> str:
        lines = []
        for turn in state.turns:
            label = self._model_label(turn.model)
            suffix = f" error={turn.error}" if turn.error else ""
            preview = turn.response[:1200] if turn.response else ""
            lines.append(f"- {label} iteration {turn.iteration} {turn.phase}:{suffix}\n{preview}")
        return "\n\n".join(lines)


_default_runner: DebateNodeRunner | None = None


def configure_default_runner(settings: dict) -> DebateNodeRunner:
    global _default_runner
    _default_runner = DebateNodeRunner(settings)
    return _default_runner


def _require_default_runner() -> DebateNodeRunner:
    if _default_runner is None:
        raise RuntimeError("Default DebateNodeRunner is not configured")
    return _default_runner


async def initialize_browser_sessions(state: DebateState) -> DebateState:
    return await _require_default_runner().initialize_browser_sessions(state)


async def initial_answer_round(state: DebateState) -> DebateState:
    return await _require_default_runner().initial_answer_round(state)


async def critique_round(state: DebateState) -> DebateState:
    return await _require_default_runner().critique_round(state)


async def refinement_round(state: DebateState) -> DebateState:
    return await _require_default_runner().refinement_round(state)


async def final_synthesis(state: DebateState) -> DebateState:
    return await _require_default_runner().final_synthesis(state)


async def log_result(state: DebateState) -> DebateState:
    return await _require_default_runner().log_result(state)
