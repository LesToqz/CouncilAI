from __future__ import annotations

import asyncio
import html
import re
from typing import Any

import gradio as gr

from src.browser.edge_manager import EdgeManager
from src.orchestration.graph import DebateGraph
from src.orchestration.state import DebateState
from src.storage.sqlite_store import SQLiteStore
from src.utils.edge_launcher import launch_controlled_edge
from src.utils.text_cleaner import clean_response_text


MODEL_ORDER = ("chatgpt", "gemini", "claude")
MODEL_ACCENTS = {
    "chatgpt": "#f4f4f5",
    "gemini": "#7aa2ff",
    "claude": "#f59e66",
}
HISTORY_LIMIT = 50


def build_app(settings: dict) -> tuple[gr.Blocks, Any, str]:
    app_name = settings.get("app", {}).get("name", "CouncilAI")
    max_iterations = int(settings.get("debate", {}).get("max_iterations", 5))
    default_iterations = int(settings.get("debate", {}).get("default_iterations", 2))
    SQLiteStore(settings).init_db()

    async def handle_check_tabs() -> str:
        return await check_existing_tabs(settings)

    async def handle_open_chatgpt() -> str:
        return await open_model_tab(settings, "chatgpt")

    async def handle_open_gemini() -> str:
        return await open_model_tab(settings, "gemini")

    async def handle_open_claude() -> str:
        return await open_model_tab(settings, "claude")

    async def handle_run(
        prompt: str,
        mode: str,
        iterations: float,
        use_chatgpt: bool,
        use_gemini: bool,
        use_claude: bool,
    ):
        async for update in run_debate_stream(
            settings,
            prompt,
            mode,
            iterations,
            use_chatgpt,
            use_gemini,
            use_claude,
        ):
            yield update

    def handle_new_chat() -> tuple[Any, ...]:
        return (
            gr.update(value="", visible=False),
            gr.update(value=""),
            _empty_status_markdown(),
            *_history_slot_updates(settings),
        )

    def handle_history_select(run_id: str | None) -> tuple[str, Any, Any]:
        return load_history_run(settings, run_id)

    def toggle_options(is_open: bool) -> tuple[bool, Any, Any]:
        next_open = not bool(is_open)
        return (
            next_open,
            gr.update(visible=next_open),
            gr.update(value="x" if next_open else "+"),
        )

    def toggle_sidebar(is_open: bool) -> tuple[bool, str, Any]:
        next_open = not bool(is_open)
        return (
            next_open,
            _sidebar_state_style(next_open),
            gr.update(value="<<" if next_open else ">>"),
        )

    theme = gr.themes.Soft(
        primary_hue="gray",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    )

    with gr.Blocks(title=app_name, elem_id="council-shell") as demo:
        drawer_open = gr.State(False)
        sidebar_open = gr.State(True)
        initial_history = _history_items(settings)
        history_buttons = []
        history_run_states = []
        sidebar_state_style = gr.HTML(
            value=_sidebar_state_style(True),
            elem_classes=["sidebar-state-style"],
        )
        sidebar_toggle_button = gr.Button("<<", elem_classes=["sidebar-toggle-button"])

        with gr.Row(elem_classes=["council-layout"]):
            with gr.Column(scale=0, min_width=0, elem_classes=["history-sidebar"]):
                new_chat_button = gr.Button("New chat", elem_classes=["history-new-chat"])
                with gr.Column(min_width=0, elem_classes=["history-list"]):
                    for index in range(HISTORY_LIMIT):
                        history_item = initial_history[index] if index < len(initial_history) else None
                        history_run_states.append(gr.State(history_item["run_id"] if history_item else None))
                        history_buttons.append(
                            gr.Button(
                                history_item["title"] if history_item else "",
                                visible=history_item is not None,
                                elem_classes=["history-item"],
                            )
                        )

            with gr.Column(elem_classes=["main-chat-area"]):
                with gr.Row(elem_classes=["topbar"]):
                    gr.HTML(
                        f"""
                        <div class="brand">
                            <div class="brand-mark"></div>
                            <div>
                                <h1>{html.escape(app_name)}</h1>
                                <p>Multi-model debate assistant</p>
                            </div>
                        </div>
                        """
                    )

                with gr.Column(elem_classes=["answer-shell"]):
                    observable_box = gr.HTML(
                        value="",
                        visible=False,
                        label="Mode Output",
                        elem_classes=["debate-trace-box"],
                    )

                with gr.Column(elem_classes=["composer-shell"]):
                    with gr.Column(visible=False, elem_classes=["options-drawer"]) as options_drawer:
                        gr.HTML(
                            """
                            <div class="drawer-header">
                                <div>
                                    <p class="eyebrow">Run Settings</p>
                                    <h2>Connections and iterations</h2>
                                </div>
                            </div>
                            """
                        )
                        status_box = gr.Markdown(
                            value=_empty_status_markdown(),
                            elem_classes=["status-box"],
                            sanitize_html=False,
                        )
                        with gr.Row(elem_classes=["drawer-grid"]):
                            with gr.Column(scale=5, elem_classes=["drawer-card"]):
                                gr.HTML('<p class="field-label">Mode</p>')
                                mode_radio = gr.Radio(
                                    choices=["Normal", "Debate"],
                                    value="Normal",
                                    label="Mode",
                                    show_label=False,
                                    container=False,
                                )
                            with gr.Column(scale=7, elem_classes=["drawer-card"]):
                                gr.HTML('<p class="field-label">Debate iterations</p>')
                                iterations_slider = gr.Slider(
                                    minimum=1,
                                    maximum=max_iterations,
                                    value=min(default_iterations, max_iterations),
                                    step=1,
                                    label="Number of Debate Iterations",
                                    show_label=False,
                                    container=False,
                                )
                        with gr.Row(elem_classes=["drawer-grid"]):
                            with gr.Column(scale=7, elem_classes=["drawer-card model-card"]):
                                chatgpt_checkbox = gr.Checkbox(value=True, label="Use ChatGPT")
                                gemini_checkbox = gr.Checkbox(value=True, label="Use Gemini")
                                claude_checkbox = gr.Checkbox(value=True, label="Use Claude")
                            with gr.Column(scale=5, elem_classes=["drawer-actions"]):
                                check_tabs_button = gr.Button("Check Connections")
                                open_chatgpt_button = gr.Button("Open ChatGPT")
                                open_gemini_button = gr.Button("Open Gemini")
                                open_claude_button = gr.Button("Open Claude")

                    with gr.Row(elem_classes=["composer-bar"]):
                        plus_button = gr.Button("+", elem_classes=["composer-icon-button"])
                        prompt_box = gr.Textbox(
                            label="Prompt",
                            lines=1,
                            show_label=False,
                            placeholder="Ask anything",
                            container=False,
                            elem_classes=["composer-input"],
                        )
                        execute_button = gr.Button("Execute", variant="primary", elem_classes=["execute-button"])

                    gr.HTML('<p class="composer-note">CouncilAI where AI brains discuss each other</p>')

        plus_button.click(
            fn=toggle_options,
            inputs=drawer_open,
            outputs=[drawer_open, options_drawer, plus_button],
        )

        sidebar_toggle_button.click(
            fn=toggle_sidebar,
            inputs=sidebar_open,
            outputs=[sidebar_open, sidebar_state_style, sidebar_toggle_button],
        )

        check_tabs_button.click(
            fn=handle_check_tabs,
            inputs=None,
            outputs=status_box,
        )

        open_chatgpt_button.click(
            fn=handle_open_chatgpt,
            inputs=None,
            outputs=status_box,
        )

        open_gemini_button.click(
            fn=handle_open_gemini,
            inputs=None,
            outputs=status_box,
        )

        open_claude_button.click(
            fn=handle_open_claude,
            inputs=None,
            outputs=status_box,
        )

        new_chat_button.click(
            fn=handle_new_chat,
            inputs=None,
            outputs=[
                observable_box,
                prompt_box,
                status_box,
                *history_buttons,
                *history_run_states,
            ],
        )

        for history_button, history_run_state in zip(history_buttons, history_run_states, strict=True):
            history_button.click(
                fn=handle_history_select,
                inputs=history_run_state,
                outputs=[status_box, observable_box, prompt_box],
            )

        mode_radio.change(
            fn=lambda mode: gr.update(visible=False),
            inputs=mode_radio,
            outputs=[
                observable_box,
            ],
        )

        execute_button.click(
            fn=handle_run,
            inputs=[
                prompt_box,
                mode_radio,
                iterations_slider,
                chatgpt_checkbox,
                gemini_checkbox,
                claude_checkbox,
            ],
            outputs=[
                status_box,
                observable_box,
                *history_buttons,
                *history_run_states,
            ],
        )

        prompt_box.submit(
            fn=handle_run,
            inputs=[
                prompt_box,
                mode_radio,
                iterations_slider,
                chatgpt_checkbox,
                gemini_checkbox,
                claude_checkbox,
            ],
            outputs=[
                status_box,
                observable_box,
                *history_buttons,
                *history_run_states,
            ],
        )

        demo.load(
            fn=lambda: _history_slot_updates(settings),
            inputs=None,
            outputs=[*history_buttons, *history_run_states],
        )

    return demo, theme, _custom_css()


async def check_existing_tabs(settings: dict) -> str:
    app_settings = settings.get("app", {})
    app_url = f"http://{app_settings.get('host', '127.0.0.1')}:{app_settings.get('port', 7860)}"
    launch_result = launch_controlled_edge(settings, app_url)
    manager = EdgeManager(settings)
    try:
        found = await manager.find_existing_ai_tabs()
        return _connection_status_html(settings, found, launch_result.message)
    except Exception as exc:  # noqa: BLE001
        return _message_panel("Attach failed", str(exc))
    finally:
        await manager.stop()


async def open_model_tab(settings: dict, model_key: str) -> str:
    app_settings = settings.get("app", {})
    app_url = f"http://{app_settings.get('host', '127.0.0.1')}:{app_settings.get('port', 7860)}"
    launch_controlled_edge(settings, app_url)
    manager = EdgeManager(settings)
    site = settings.get("model_sites", {}).get(model_key, {})
    site_url = site.get("url")
    site_name = site.get("name", model_key.title())
    if not site_url:
        return _message_panel("Open failed", f"No URL configured for {site_name}.")

    try:
        await manager.open_url_in_background(site_url, app_url=app_url)
        found = await manager.find_existing_ai_tabs()
        return _connection_status_html(settings, found, f"{site_name} tab opened. CouncilAI should stay in front.")
    except Exception as exc:  # noqa: BLE001
        return _message_panel("Open failed", str(exc))
    finally:
        await manager.stop()


def _connection_status_html(settings: dict, found: dict[str, str | None], message: str) -> str:
    statuses = [f'<p class="status-message">{html.escape(message)}</p>', '<div class="connection-list">']
    for model_key in MODEL_ORDER:
        site = settings["model_sites"][model_key]
        if found.get(model_key):
            statuses.append(f"<p><strong>{html.escape(site['name'])}</strong><span>Connected</span></p>")
        else:
            statuses.append(f"<p><strong>{html.escape(site['name'])}</strong><span>Missing</span></p>")
    statuses.append("</div>")
    return "\n".join(statuses)


def _history_items(settings: dict) -> list[dict[str, Any]]:
    return SQLiteStore(settings).list_chat_history(HISTORY_LIMIT)


def _history_slot_updates(settings: dict) -> tuple[Any, ...]:
    items = _history_items(settings)
    button_updates = []
    run_ids: list[str | None] = []
    for index in range(HISTORY_LIMIT):
        item = items[index] if index < len(items) else None
        button_updates.append(
            gr.update(
                value=item["title"] if item else "",
                visible=item is not None,
            )
        )
        run_ids.append(item["run_id"] if item else None)
    return (*button_updates, *run_ids)


def _sidebar_state_style(is_open: bool) -> str:
    if is_open:
        return """
<style>
:root { --history-sidebar-width: 260px; }
.sidebar-toggle-button { left: calc(var(--history-sidebar-width) + 18px) !important; }
.topbar .brand { padding-left: 52px !important; }
@media (max-width: 760px) {
    :root { --history-sidebar-width: 220px; }
    .sidebar-toggle-button { left: calc(var(--history-sidebar-width) + 12px) !important; }
}
</style>
"""

    return """
<style>
:root { --history-sidebar-width: 0px; }
.history-sidebar {
    transform: translateX(-110%) !important;
    opacity: 0 !important;
    pointer-events: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    border-right: 0 !important;
}
.sidebar-toggle-button { left: 18px !important; }
.topbar .brand { padding-left: 52px !important; }
.debate-view-toggle { left: 22px !important; }
@media (max-width: 760px) {
    :root { --history-sidebar-width: 0px; }
    .sidebar-toggle-button { left: 12px !important; }
    .debate-view-toggle { left: 12px !important; }
}
</style>
"""


def load_history_run(settings: dict, run_id: str | None) -> tuple[str, Any, Any, Any]:
    if not run_id:
        return (
            _empty_status_markdown(),
            gr.update(value="", visible=False),
            gr.update(value=""),
        )

    store = SQLiteStore(settings)
    detail = store.get_run_detail(run_id)
    if detail is None:
        return (
            _message_panel("History unavailable", "This saved run could not be found."),
            gr.update(value="", visible=False),
            gr.update(value=""),
        )

    state = store.state_from_detail(detail)
    selected_models = state.active_models or list(MODEL_ORDER)
    if state.mode == "debate":
        compare_html = _observable_board_html(settings, state, [], selected_models)
        output_html = (
            _debate_switcher_html(compare_html, _final_answer_html(state.final_answer), state.run_id)
            if state.final_answer
            else compare_html
        )
    else:
        output_html = _normal_board_html(settings, state, selected_models)

    return (
        _history_status_html(detail["run"]),
        gr.update(value=output_html, visible=True),
        gr.update(value=state.user_prompt),
    )


async def run_debate(
    settings: dict,
    prompt: str,
    mode_label: str,
    iterations: float,
    use_chatgpt: bool,
    use_gemini: bool,
    use_claude: bool,
) -> tuple[Any, ...]:
    final_update = (
        "",
        gr.update(value="", visible=False),
        *_history_slot_updates(settings),
    )
    async for update in run_debate_stream(
        settings,
        prompt,
        mode_label,
        iterations,
        use_chatgpt,
        use_gemini,
        use_claude,
    ):
        final_update = update
    return final_update


async def run_debate_stream(
    settings: dict,
    prompt: str,
    mode_label: str,
    iterations: float,
    use_chatgpt: bool,
    use_gemini: bool,
    use_claude: bool,
):
    prompt = (prompt or "").strip()
    active_models = _selected_models(use_chatgpt, use_gemini, use_claude)
    mode = "debate" if mode_label == "Debate" else "normal"

    if not prompt:
        yield (
            _message_panel("Prompt required", "Enter a prompt before running."),
            gr.update(value="", visible=False),
            *_history_slot_updates(settings),
        )
        return

    min_models = 2 if mode == "debate" else 1
    if len(active_models) < min_models:
        yield (
            _message_panel("Model selection", f"Select at least {min_models} model{'' if min_models == 1 else 's'}."),
            gr.update(value="", visible=False),
            *_history_slot_updates(settings),
        )
        return

    max_iterations = int(settings.get("debate", {}).get("max_iterations", 5))
    iteration_count = max(1, min(int(iterations), max_iterations))

    state = DebateState(
        user_prompt=prompt,
        mode=mode,
        max_iterations=iteration_count,
        active_models=active_models,
    )
    SQLiteStore(settings).save_run_summary(state, status="running")

    progress_messages = ["Normal run queued" if mode == "normal" else "Debate queued"]
    progress_queue: asyncio.Queue[tuple[str, DebateState | None]] = asyncio.Queue()

    async def on_progress(message: str, progress_state: DebateState | None = None) -> None:
        progress_messages.append(message)
        del progress_messages[:-14]
        progress_queue.put_nowait((message, progress_state))

    if mode == "debate":
        compare_html = _observable_board_html(settings, state, progress_messages, active_models)
        yield (
            _live_status_markdown(active_models, progress_messages, mode),
            gr.update(value=compare_html, visible=True),
            *_history_slot_updates(settings),
        )
    else:
        compare_html = _normal_board_html(settings, state, active_models)
        yield (
            _live_status_markdown(active_models, progress_messages, mode),
            gr.update(value=compare_html, visible=True),
            *_history_slot_updates(settings),
        )

    graph = DebateGraph(settings, progress_callback=on_progress)
    latest_state = state
    task = asyncio.create_task(graph.ainvoke(state))
    try:
        while not task.done():
            try:
                _, progress_state = await asyncio.wait_for(progress_queue.get(), timeout=0.35)
            except asyncio.TimeoutError:
                continue

            if progress_state is not None:
                latest_state = progress_state

            if mode == "debate":
                compare_html = _observable_board_html(settings, latest_state, progress_messages, active_models)
                output_html = (
                    _debate_switcher_html(
                        compare_html,
                        _final_answer_html(latest_state.final_answer),
                        latest_state.run_id,
                    )
                    if latest_state.final_answer
                    else compare_html
                )
                yield (
                    _live_status_markdown(latest_state.active_models or active_models, progress_messages, mode),
                    gr.update(
                        value=output_html,
                        visible=True,
                    ),
                    *_history_slot_updates(settings),
                )
            else:
                compare_html = _normal_board_html(settings, latest_state, active_models)
                yield (
                    _live_status_markdown(latest_state.active_models or active_models, progress_messages, mode),
                    gr.update(value=compare_html, visible=True),
                    *_history_slot_updates(settings),
                )

        final_state = await task
    except Exception as exc:  # noqa: BLE001 - UI must show unexpected orchestration failures.
        latest_state.errors.append(str(exc))
        SQLiteStore(settings).save_run_summary(latest_state, status="failed")
        yield (
            _message_panel("Run failed", str(exc)),
            gr.update(value="", visible=False),
            *_history_slot_updates(settings),
        )
        return

    status = _status_markdown(final_state)
    if mode == "normal":
        compare_html = _normal_board_html(settings, final_state, active_models)
        yield (
            status,
            gr.update(value=compare_html, visible=True),
            *_history_slot_updates(settings),
        )
        return

    compare_html = _observable_board_html(settings, final_state, progress_messages, active_models)
    final_html = _final_answer_html(final_state.final_answer)
    yield (
        status,
        gr.update(value=_debate_switcher_html(compare_html, final_html, final_state.run_id), visible=True),
        *_history_slot_updates(settings),
    )


def _selected_models(use_chatgpt: bool, use_gemini: bool, use_claude: bool) -> list[str]:
    selected = []
    if use_chatgpt:
        selected.append("chatgpt")
    if use_gemini:
        selected.append("gemini")
    if use_claude:
        selected.append("claude")
    return selected


def _status_markdown(state: DebateState) -> str:
    active_models = ", ".join(model.upper() for model in state.active_models) or "None"
    lines = [
        '<div class="status-grid">',
        _stat_card("Run ID", state.run_id),
        _stat_card("Iterations", str(state.current_iteration)),
        _stat_card("Active Models", active_models),
        _stat_card("Turns Logged", str(len(state.turns))),
        "</div>",
    ]
    if state.errors:
        warnings = "".join(f"<li>{html.escape(error)}</li>" for error in state.errors)
        lines.append(f'<div class="warning-list"><strong>Warnings</strong><ul>{warnings}</ul></div>')
    return "\n".join(lines)


def _history_status_html(run: dict[str, Any]) -> str:
    return (
        '<div class="status-grid">'
        f'{_stat_card("Loaded", run.get("title") or "Saved chat")}'
        f'{_stat_card("Mode", (run.get("mode") or "normal").title())}'
        f'{_stat_card("Status", run.get("status") or "saved")}'
        f'{_stat_card("Created", run.get("created_at") or "unknown")}'
        "</div>"
    )


def _history_summary_html(detail: dict[str, Any]) -> str:
    run = detail["run"]
    prompt = html.escape(run.get("user_prompt") or "")
    final_answer = clean_response_text(run.get("final_answer") or "")
    errors = [turn.get("error") for turn in detail.get("turns", []) if turn.get("error")]
    parts = [
        '<section class="history-loaded-summary">',
        "<h2>Saved chat</h2>",
        '<p class="eyebrow">Original prompt</p>',
        f"<blockquote>{prompt}</blockquote>",
    ]
    if final_answer:
        parts.extend(
            [
                '<p class="eyebrow">Final answer</p>',
                _markdown_fragment_to_html(final_answer),
            ]
        )
    if errors:
        parts.append('<div class="warning-list"><strong>Errors</strong><ul>')
        parts.extend(f"<li>{html.escape(error or '')}</li>" for error in errors)
        parts.append("</ul></div>")
    parts.append("</section>")
    return "\n".join(parts)


def _live_status_markdown(active_models: list[str], messages: list[str], mode: str) -> str:
    active = ", ".join(model.upper() for model in active_models) or "None"
    latest = html.escape(messages[-1] if messages else "Waiting")
    mode_title = "Debate" if mode == "debate" else "Normal"
    return (
        '<div class="status-grid">'
        f'{_stat_card("Status", latest)}'
        f'{_stat_card("Active Models", active)}'
        f'{_stat_card("Events", str(len(messages)))}'
        f'{_stat_card("Mode", mode_title)}'
        "</div>"
    )


def _normal_board_html(settings: dict, state: DebateState, selected_models: list[str]) -> str:
    columns = "\n".join(
        _model_column_html(
            settings,
            state,
            model_key,
            progress_messages=[],
            selected_models=selected_models,
            output_only=True,
        )
        for model_key in MODEL_ORDER
    )
    return (
        '<section class="debate-board normal-board">'
        f'<div class="debate-columns">{columns}</div>'
        "</section>"
    )


def _observable_board_html(
    settings: dict,
    state: DebateState,
    progress_messages: list[str],
    selected_models: list[str],
) -> str:
    del progress_messages
    columns = "\n".join(
        _model_column_html(settings, state, model_key, [], selected_models)
        for model_key in MODEL_ORDER
    )
    return (
        '<section class="debate-board">'
        f'<div class="debate-columns">{columns}</div>'
        "</section>"
    )


def _model_column_html(
    settings: dict,
    state: DebateState,
    model_key: str,
    progress_messages: list[str],
    selected_models: list[str],
    output_only: bool = False,
) -> str:
    label = _model_label(settings, model_key)
    accent = MODEL_ACCENTS.get(model_key, "#f4f4f5")
    turns = [turn for turn in state.turns if turn.model == model_key]
    status_label, status_class = _model_status(label, model_key, state, turns, progress_messages, selected_models)
    visible_turns = [turn for turn in turns if turn.phase == "initial_answer"] if output_only else [
        turn for turn in turns if turn.phase != "initialize"
    ]
    cards = "\n".join(_turn_card_html(turn, output_only=output_only) for turn in visible_turns)
    if not cards:
        cards = (
            '<div class="empty-turn">'
            "<strong>Waiting for output</strong>"
            f"<p>{'This model output will appear here.' if output_only else 'This model will show its answer, critique, refinement, and synthesis attempts here.'}</p>"
            "</div>"
        )

    return (
        f'<article class="debate-column {status_class}" style="--model-accent: {accent};">'
        '<div class="debate-model-header">'
        f'<div class="model-orb">{html.escape(label[:1])}</div>'
        '<div>'
        f'<h3>{html.escape(label)}</h3>'
        f'<p>{html.escape(_model_role(settings, model_key))}</p>'
        "</div>"
        f'<span class="model-status">{html.escape(status_label)}</span>'
        "</div>"
        '<div class="debate-column-body">'
        f"{cards}"
        "</div>"
        "</article>"
    )


def _turn_card_html(turn, output_only: bool = False) -> str:
    phase = "Output" if output_only else _phase_label(turn.phase)
    iteration = "" if output_only else (
        "setup" if turn.iteration == 0 and turn.phase == "initialize" else f"iteration {turn.iteration}"
    )
    if turn.error:
        body = f'<div class="turn-error">{html.escape(turn.error)}</div>'
    elif turn.response:
        body = _markdown_fragment_to_html(turn.response)
    else:
        body = '<p class="muted-copy">No output captured.</p>'

    iteration_html = f'<span>{html.escape(iteration)}</span>' if iteration else ""
    return (
        '<section class="turn-card">'
        '<div class="turn-meta">'
        f'<span>{html.escape(phase)}</span>'
        f"{iteration_html}"
        "</div>"
        f'<div class="turn-content">{body}</div>'
        "</section>"
    )


def _model_status(
    label: str,
    model_key: str,
    state: DebateState,
    turns: list,
    progress_messages: list[str],
    selected_models: list[str],
) -> tuple[str, str]:
    if model_key not in selected_models:
        return "Not selected", "is-muted"

    latest_model_event = next(
        (message for message in reversed(progress_messages) if message.startswith(f"{label}:")),
        "",
    )
    if "failed" in latest_model_event:
        return "Needs attention", "has-error"
    if "started" in latest_model_event or "checking" in latest_model_event:
        return "Working", "is-active"
    if "complete" in latest_model_event:
        return "Updated", "is-complete"
    if model_key in state.active_models:
        return "Connected", "is-complete" if turns else "is-active"
    if turns and any(turn.error for turn in turns):
        return "Error", "has-error"
    return "Waiting", "is-muted"


def _phase_label(phase: str) -> str:
    labels = {
        "initial_answer": "Initial answer",
        "critique": "Critique",
        "refinement": "Refinement",
        "final_synthesis": "Final synthesis",
        "initialize": "Connection",
    }
    return labels.get(phase, phase.replace("_", " ").title())


def _model_label(settings: dict, model_key: str) -> str:
    return settings.get("model_sites", {}).get(model_key, {}).get("name", model_key.title())


def _model_role(settings: dict, model_key: str) -> str:
    return settings.get("model_sites", {}).get(model_key, {}).get("role", "council participant")


def _markdown_fragment_to_html(text: str) -> str:
    lines = clean_response_text(text).splitlines()
    parts: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if _is_table_start(lines, index):
            table_lines = []
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            parts.append(_markdown_table_to_html(table_lines))
            continue

        if line.startswith("#"):
            depth = min(len(line) - len(line.lstrip("#")), 4)
            content = line[depth:].strip()
            tag = "h4" if depth >= 3 else "h3"
            parts.append(f"<{tag}>{_inline_markdown(content)}</{tag}>")
            index += 1
            continue

        if _is_bullet(line):
            items = []
            while index < len(lines) and _is_bullet(lines[index].strip()):
                items.append(lines[index].strip()[2:].strip())
                index += 1
            parts.append("<ul>" + "".join(f"<li>{_inline_markdown(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith("#") or _is_bullet(candidate) or _is_table_start(lines, index):
                break
            paragraph.append(candidate)
            index += 1
        parts.append(f"<p>{_inline_markdown(' '.join(paragraph))}</p>")

    return "\n".join(parts) if parts else '<p class="muted-copy">No response text captured.</p>'


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def _is_bullet(line: str) -> bool:
    return line.startswith("- ") or line.startswith("* ") or line.startswith("• ")


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index].strip()
    next_line = lines[index + 1].strip()
    return "|" in current and "|" in next_line and set(next_line.replace("|", "").replace(":", "").strip()) <= {"-", " "}


def _markdown_table_to_html(table_lines: list[str]) -> str:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table_lines
        if line.strip()
    ]
    if len(rows) < 2:
        return "<p>" + _inline_markdown(" ".join(table_lines)) + "</p>"

    header = rows[0]
    body_rows = rows[2:] if _is_delimiter_row(rows[1]) else rows[1:]
    header_html = "".join(f"<th>{_inline_markdown(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_inline_markdown(cell)}</td>" for cell in row) + "</tr>"
        for row in body_rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _is_delimiter_row(row: list[str]) -> bool:
    return all(set(cell.replace(":", "").strip()) <= {"-"} for cell in row)


def _format_final_answer(answer: str | None) -> str:
    return clean_response_text(answer or "")


def _final_answer_html(answer: str | None) -> str:
    return (
        '<section class="final-only-view">'
        f"{_markdown_fragment_to_html(answer or '')}"
        "</section>"
    )


def _debate_switcher_html(compare_html: str, final_html: str, run_id: str) -> str:
    toggle_id = f"debate-view-toggle-{re.sub(r'[^a-zA-Z0-9_-]', '', run_id)}"
    return (
        '<section class="debate-switcher">'
        f'<input id="{html.escape(toggle_id)}" class="debate-view-checkbox" type="checkbox">'
        f'<div class="debate-compare-view">{compare_html}</div>'
        f'<div class="debate-final-view">{final_html}</div>'
        f'<label for="{html.escape(toggle_id)}" class="debate-view-toggle">'
        '<span class="toggle-final-label">final</span>'
        '<span class="toggle-compare-label">compare</span>'
        "</label>"
        "</section>"
    )


def _stat_card(label: str, value: str) -> str:
    return (
        '<div class="stat-card">'
        f'<p class="stat-label">{html.escape(label)}</p>'
        f'<p class="stat-value">{html.escape(value)}</p>'
        "</div>"
    )


def _message_panel(title: str, message: str) -> str:
    return (
        '<div class="warning-list">'
        f"<strong>{html.escape(title)}</strong>"
        f"<p>{html.escape(message)}</p>"
        "</div>"
    )


def _empty_status_markdown() -> str:
    return (
        '<div class="connection-list">'
        "<p><strong>ChatGPT</strong><span>Not checked</span></p>"
        "<p><strong>Gemini</strong><span>Not checked</span></p>"
        "<p><strong>Claude</strong><span>Not checked</span></p>"
        "</div>"
    )


def _welcome_markdown() -> str:
    return (
        "# CouncilAI\n\n"
        "Ask a question below. Use the plus button to check model connections, choose debate mode, "
        "and set the number of iterations."
    )


def _custom_css() -> str:
    return """
:root {
    --cai-bg: #000000;
    --cai-panel: #111111;
    --cai-panel-2: #1f1f1f;
    --cai-panel-3: #2a2a2a;
    --cai-text: #f4f4f5;
    --cai-muted: #a3a3a3;
    --cai-line: #2f2f2f;
    --cai-line-strong: #4a4a4a;
    --cai-button: #ffffff;
    --cai-button-text: #111111;
    --cai-warn-bg: #2d2012;
    --cai-warn-line: #7c4a16;
    --cai-warn-text: #fed7aa;
    --cai-topbar-height: 96px;
    --cai-composer-height: 138px;
    --history-sidebar-width: 260px;
}

html,
body,
#root,
.gradio-container,
.main,
.contain {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    margin: 0 !important;
    background: var(--cai-bg) !important;
    color: var(--cai-text) !important;
}
html,
body,
#root,
.gradio-container,
.main {
    height: 100vh !important;
    overflow: hidden !important;
}
footer { display: none !important; visibility: hidden !important; }
.gradio-container,
.gradio-container .main,
.gradio-container .main.fillable {
    min-height: 100vh !important;
    padding: 0 !important;
}
#council-shell {
    min-height: 100vh !important;
    padding: 0 !important;
    background: var(--cai-bg) !important;
    overflow: visible !important;
}
.sidebar-state-style {
    display: none !important;
}
.sidebar-toggle-button,
.sidebar-toggle-button button {
    position: fixed !important;
    top: 18px !important;
    z-index: 130 !important;
    width: 40px !important;
    min-width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    border-radius: 999px !important;
    color: var(--cai-text) !important;
    background: #181818 !important;
    border: 1px solid #343434 !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38) !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}
.sidebar-toggle-button:hover,
.sidebar-toggle-button button:hover {
    background: #242424 !important;
    border-color: #4a4a4a !important;
}

/* --- Row that holds sidebar + main --- */
.council-layout {
    min-width: 0 !important;
    flex-wrap: nowrap !important;
    overflow: visible !important;
    height: 100vh !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    gap: 0 !important;
    padding: 0 !important;
}
.main-chat-area {
    min-width: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* --- Fixed left sidebar --- */
.history-sidebar {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    bottom: 0 !important;
    z-index: 90 !important;
    width: var(--history-sidebar-width) !important;
    min-width: var(--history-sidebar-width) !important;
    max-width: var(--history-sidebar-width) !important;
    height: 100vh !important;
    flex: 0 0 var(--history-sidebar-width) !important;
    padding: 14px 10px !important;
    background: #0f0f0f !important;
    border-right: 1px solid #202020 !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
}
.history-sidebar,
.history-sidebar * {
    color: var(--cai-text) !important;
}
/* Override all Gradio internal wrappers inside the sidebar */
.history-sidebar > .wrap,
.history-sidebar > .column-wrap,
.history-sidebar .wrap,
.history-sidebar .form,
.history-sidebar .block,
.history-sidebar [class*="wrap"],
.history-sidebar [class*="block"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    min-width: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
    min-height: 0 !important;
    height: auto !important;
    flex: 1 1 auto !important;
}
/* But the sidebar root itself needs fixed height */
.history-sidebar {
    height: 100vh !important;
}
.history-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-height: 42px;
    margin: auto 4px 2px;
    padding-top: 14px;
    border-top: 1px solid #202020;
    flex-shrink: 0;
}
.history-brand h2 {
    color: var(--cai-text) !important;
    font-size: 17px !important;
    line-height: 1.15 !important;
    margin: 0 !important;
}
.history-brand p {
    color: var(--cai-muted) !important;
    font-size: 11px !important;
    line-height: 1.2 !important;
    margin: 2px 0 0 !important;
}
.history-new-chat,
.history-new-chat button {
    width: 100% !important;
    min-height: 42px !important;
    border-radius: 10px !important;
    color: var(--cai-text) !important;
    background: #1b1b1b !important;
    border: 1px solid #303030 !important;
    font-size: 14px !important;
    font-weight: 650 !important;
    flex-shrink: 0 !important;
    margin-bottom: 6px !important;
}
.history-new-chat:hover,
.history-new-chat button:hover {
    background: #242424 !important;
    border-color: #3a3a3a !important;
}

/* --- Scrollable history list inside sidebar --- */
.history-list {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    margin-top: 10px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-color: #383838 #0f0f0f;
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    padding: 0 !important;
}
/* Gradio wraps Column children in an inner div — make it scrollable too */
.history-list > .wrap,
.history-list > .column-wrap,
.history-list > [class*="wrap"] {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    scrollbar-color: #383838 #0f0f0f;
}
.history-list .history-item,
.history-list .history-item button,
.history-item,
.history-item button {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    min-height: 34px !important;
    margin: 0 !important;
    padding: 0 10px !important;
    border-radius: 9px !important;
    color: #d9d9d9 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    font-size: 13px !important;
    line-height: 1.25 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}
.history-list .history-item:hover,
.history-list .history-item:hover button,
.history-item:hover,
.history-item:hover button {
    background: #1c1c1c !important;
}
.history-item-active {
    color: var(--cai-text) !important;
    background: #242424 !important;
    border-color: #333333 !important;
}
.main-chat-area {
    margin-left: var(--history-sidebar-width) !important;
    width: calc(100vw - var(--history-sidebar-width)) !important;
}

.topbar {
    position: fixed;
    top: 0;
    left: var(--history-sidebar-width);
    right: 0;
    z-index: 70;
    width: calc(100% - var(--history-sidebar-width)) !important;
    min-height: var(--cai-topbar-height);
    margin: 0 !important;
    padding: 12px 28px !important;
    border: 0 !important;
    border-bottom: 1px solid #151515 !important;
    border-radius: 0 !important;
    background: rgba(0, 0, 0, 0.92) !important;
    backdrop-filter: blur(14px);
    box-shadow: none !important;
}
.topbar > .wrap,
.topbar > .form,
.topbar .block {
    width: 100% !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.brand {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 16px;
    width: 100%;
}
.brand > div:last-child {
    margin-right: auto;
}
.brand-mark {
    width: 22px;
    height: 22px;
    border: 1.5px solid var(--cai-text);
    border-radius: 7px;
    position: relative;
}
.brand-mark::before {
    content: "";
    position: absolute;
    left: 4px;
    top: -5px;
    width: 10px;
    height: 7px;
    border: 1.5px solid var(--cai-text);
    border-bottom: 0;
    border-radius: 5px 5px 0 0;
    background: var(--cai-bg);
}
.brand h1 {
    margin: 0;
    color: var(--cai-text);
    font-size: 23px;
    font-weight: 650;
    letter-spacing: 0;
    line-height: 1.1;
}
.brand p {
    margin: 2px 0 0;
    color: var(--cai-muted);
    font-size: 12px;
}
.answer-shell {
    position: fixed;
    left: var(--history-sidebar-width);
    right: 0;
    top: var(--cai-topbar-height);
    bottom: var(--cai-composer-height);
    z-index: 1;
    width: calc(100% - var(--history-sidebar-width)) !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 26px 18px 70px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    scrollbar-color: #444444 #000000;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.answer-shell::after {
    content: "";
    position: fixed;
    left: var(--history-sidebar-width);
    right: 0;
    bottom: var(--cai-composer-height);
    height: 86px;
    z-index: 5;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.82) 72%, #000 100%);
}
.answer-shell .block,
.answer-shell .wrap,
.answer-shell .form,
.answer-shell .container,
.answer-shell [class*="block"],
.answer-shell [class*="wrap"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.final-answer-box,
.debate-trace-box {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--cai-text) !important;
}
.history-loaded-summary {
    color: var(--cai-text);
}
.history-loaded-summary blockquote {
    margin: 10px 0 24px !important;
    padding: 12px 14px !important;
    color: #e6e6e6 !important;
    background: #151515 !important;
    border: 1px solid #2f2f2f !important;
    border-left: 3px solid #666666 !important;
    border-radius: 10px !important;
}
.final-answer-box {
    max-width: 960px;
    margin: 0 auto !important;
    padding: 0 0 20px !important;
}
.debate-trace-box {
    width: min(100%, 1540px) !important;
    max-width: none;
    margin: 22px auto 0 !important;
    padding: 8px 0 0 !important;
    border-top: 1px solid var(--cai-line) !important;
}
.final-answer-box h1,
.final-answer-box h2,
.final-answer-box h3,
.debate-trace-box h1,
.debate-trace-box h2,
.debate-trace-box h3 {
    color: var(--cai-text) !important;
    letter-spacing: 0 !important;
}
.final-answer-box h1 {
    font-size: 30px !important;
    font-weight: 650 !important;
    line-height: 1.25 !important;
    margin: 0 0 22px !important;
}
.final-answer-box h2,
.debate-trace-box h2 {
    font-size: 24px !important;
    font-weight: 650 !important;
    line-height: 1.3 !important;
    margin: 34px 0 14px !important;
}
.final-answer-box h3,
.debate-trace-box h3 {
    font-size: 20px !important;
    font-weight: 650 !important;
    line-height: 1.35 !important;
    margin: 26px 0 10px !important;
}
.final-answer-box p,
.final-answer-box li,
.debate-trace-box p,
.debate-trace-box li {
    color: #ececec !important;
    font-size: 17px !important;
    line-height: 1.7 !important;
}
.final-answer-box p,
.debate-trace-box p {
    margin: 0 0 18px !important;
}
.final-answer-box ul,
.final-answer-box ol,
.debate-trace-box ul,
.debate-trace-box ol {
    margin: 10px 0 22px 28px !important;
    padding: 0 !important;
}
.final-answer-box li,
.debate-trace-box li {
    margin: 6px 0 !important;
}
.final-answer-box a,
.debate-trace-box a,
.status-box a {
    color: #8ab4ff !important;
}
.final-answer-box img,
.final-answer-box picture,
.final-answer-box svg {
    display: none !important;
}
.final-answer-box table,
.debate-trace-box table {
    display: table !important;
    width: 100% !important;
    margin: 22px 0 28px !important;
    border-collapse: collapse !important;
    overflow: hidden !important;
    color: var(--cai-text) !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}
.final-answer-box thead,
.debate-trace-box thead {
    background: #171717 !important;
}
.final-answer-box th,
.final-answer-box td,
.debate-trace-box th,
.debate-trace-box td {
    padding: 12px 14px !important;
    border-bottom: 1px solid var(--cai-line) !important;
    text-align: left !important;
    vertical-align: top !important;
}
.final-answer-box th,
.debate-trace-box th {
    color: var(--cai-text) !important;
    font-weight: 700 !important;
}
.final-answer-box td,
.debate-trace-box td {
    color: #e5e5e5 !important;
}
.final-answer-box blockquote,
.debate-trace-box blockquote {
    margin: 20px 0 !important;
    padding: 0 0 0 18px !important;
    border-left: 3px solid #555555 !important;
    color: #d4d4d4 !important;
}
.final-answer-box pre,
.debate-trace-box pre {
    background: #171717 !important;
    border: 1px solid var(--cai-line) !important;
    color: var(--cai-text) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
details {
    background: #101010;
    border: 1px solid var(--cai-line);
    border-radius: 10px;
    margin: 10px 0;
    overflow: hidden;
}
details summary {
    cursor: pointer;
    color: var(--cai-text);
    background: #191919;
    padding: 12px 14px;
    font-weight: 600;
}
details > *:not(summary) {
    padding: 0 14px 14px;
}

.debate-board {
    width: 100%;
    color: var(--cai-text);
}
.final-only-view {
    width: min(100%, 960px);
    margin: 0 auto;
    padding: 6px 0 36px;
    color: var(--cai-text);
}
.debate-board-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: 0 0 12px;
}
.debate-board-header h2 {
    color: var(--cai-text) !important;
    font-size: 22px !important;
    line-height: 1.25 !important;
    margin: 0 !important;
}
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #d4d4d4;
    border: 1px solid var(--cai-line);
    border-radius: 999px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 650;
}
.live-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #22c55e;
    box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.12);
}
.debate-event-strip {
    display: flex;
    gap: 8px;
    margin: 0 0 12px !important;
    padding: 0 0 2px !important;
    overflow-x: auto;
    list-style: none;
}
.debate-event-strip li {
    flex: 0 0 auto;
    max-width: 260px;
    padding: 7px 10px;
    color: #d8d8d8 !important;
    background: #151515;
    border: 1px solid #2d2d2d;
    border-radius: 999px;
    font-size: 12px !important;
    line-height: 1.2 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.debate-columns {
    display: grid;
    grid-template-columns: repeat(3, minmax(330px, 1fr));
    gap: 12px;
    width: 100%;
    overflow-x: auto;
    padding-bottom: 8px;
}
.debate-column {
    min-width: 330px;
    background: #141414;
    border: 1px solid #2b2b2b;
    border-top: 2px solid var(--model-accent);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
}
.debate-column.is-muted {
    opacity: 0.62;
}
.debate-column.is-active {
    border-color: #3b3b3b;
    box-shadow: 0 18px 50px rgba(255, 255, 255, 0.04);
}
.debate-column.has-error {
    border-color: #7c2d12;
    border-top-color: #fb923c;
}
.debate-model-header {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: 10px;
    padding: 13px 14px;
    background: #1b1b1b;
    border-bottom: 1px solid #2b2b2b;
}
.model-orb {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    color: #111111;
    background: var(--model-accent);
    border-radius: 999px;
    font-weight: 800;
}
.debate-model-header h3 {
    color: var(--cai-text) !important;
    font-size: 15px !important;
    line-height: 1.25 !important;
    margin: 0 !important;
}
.debate-model-header p {
    color: var(--cai-muted) !important;
    font-size: 11px !important;
    line-height: 1.25 !important;
    margin: 2px 0 0 !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.model-status {
    align-self: start;
    color: #d4d4d4;
    background: #242424;
    border: 1px solid #383838;
    border-radius: 999px;
    padding: 5px 8px;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
}
.debate-column-body {
    height: min(62vh, 680px);
    overflow-y: auto;
    padding: 12px;
    scrollbar-color: #444444 #151515;
}
.turn-card {
    background: #1d1d1d;
    border: 1px solid #303030;
    border-radius: 13px;
    margin: 0 0 12px;
    overflow: hidden;
}
.turn-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 11px;
    color: #d9d9d9;
    background: #252525;
    border-bottom: 1px solid #333333;
    font-size: 12px;
    font-weight: 750;
}
.turn-meta span:last-child {
    color: var(--cai-muted);
    font-weight: 600;
}
.turn-content {
    padding: 11px;
}
.turn-content h3,
.turn-content h4 {
    color: var(--cai-text) !important;
    font-size: 15px !important;
    line-height: 1.35 !important;
    margin: 12px 0 8px !important;
}
.turn-content p,
.turn-content li {
    color: #e5e5e5 !important;
    font-size: 14px !important;
    line-height: 1.58 !important;
}
.turn-content p {
    margin: 0 0 10px !important;
}
.turn-content ul,
.turn-content ol {
    margin: 6px 0 12px 20px !important;
    padding: 0 !important;
}
.turn-content table {
    width: 100% !important;
    margin: 10px 0 12px !important;
    border-collapse: collapse !important;
    font-size: 12px !important;
}
.turn-content th,
.turn-content td {
    padding: 8px !important;
    border-bottom: 1px solid #353535 !important;
    text-align: left !important;
    vertical-align: top !important;
}
.turn-content th {
    color: #f4f4f5 !important;
    background: #262626;
}
.turn-content code {
    color: #f5f5f5;
    background: #111111;
    border: 1px solid #333333;
    border-radius: 5px;
    padding: 1px 5px;
}
.turn-content pre {
    white-space: pre-wrap;
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 9px !important;
    padding: 10px !important;
}
.turn-error {
    color: var(--cai-warn-text);
    background: var(--cai-warn-bg);
    border: 1px solid var(--cai-warn-line);
    border-radius: 9px;
    padding: 9px 10px;
    font-size: 13px;
    line-height: 1.45;
}
.empty-turn {
    padding: 16px;
    color: #d6d6d6;
    background: #1d1d1d;
    border: 1px dashed #3a3a3a;
    border-radius: 13px;
}
.empty-turn strong {
    display: block;
    color: var(--cai-text);
    font-size: 14px;
    margin-bottom: 6px;
}
.empty-turn p,
.muted-copy {
    color: var(--cai-muted) !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

.composer-shell {
    position: fixed;
    left: var(--history-sidebar-width);
    right: 0;
    bottom: 0;
    z-index: 60;
    display: flex;
    align-items: center;
    width: calc(100% - var(--history-sidebar-width)) !important;
    padding: 8px 18px 8px !important;
    background: linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,0.92) 34%, #000 100%) !important;
    border: 0 !important;
    box-shadow: none !important;
}
.composer-shell .block,
.composer-shell .wrap,
.composer-shell .form,
.composer-shell .container,
.composer-shell [class*="block"],
.composer-shell [class*="wrap"],
.composer-shell [class*="form"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.composer-bar {
    width: min(100%, 1060px) !important;
    min-height: 58px;
    margin: 0 auto !important;
    padding: 7px 10px !important;
    display: flex !important;
    align-items: center !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    background: #242424 !important;
    border: 1px solid #383838 !important;
    border-radius: 28px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
}
.composer-icon-button,
.execute-button {
    flex: 0 0 auto !important;
}
.composer-icon-button button,
.composer-icon-button {
    width: 40px !important;
    min-width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    border-radius: 999px !important;
    font-size: 24px !important;
    font-weight: 300 !important;
    line-height: 1 !important;
    padding: 0 !important;
    color: #e7e7e7 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}
.composer-icon-button:hover,
.composer-icon-button button:hover {
    background: #303030 !important;
    border-color: #474747 !important;
}
.execute-button,
.execute-button button,
button.primary.execute-button {
    min-width: 104px !important;
    height: 40px !important;
    min-height: 40px !important;
    border-radius: 999px !important;
    background: var(--cai-button) !important;
    border: 1px solid var(--cai-button) !important;
    color: var(--cai-button-text) !important;
    font-weight: 700 !important;
}
.execute-button *,
.execute-button span,
button.primary.execute-button * {
    color: var(--cai-button-text) !important;
}
.composer-input {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: auto !important;
    margin: 0 !important;
}
.composer-input.block,
.composer-input.auto-margin {
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 0 !important;
    margin: 0 !important;
}
.composer-input textarea,
.composer-input input,
#council-shell textarea {
    width: 100% !important;
    min-height: 40px !important;
    max-height: 120px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--cai-text) !important;
    font-size: 17px !important;
    line-height: 1.45 !important;
    padding: 8px 2px !important;
    resize: none !important;
}
.composer-input textarea::placeholder {
    color: #b5b5b5 !important;
}
.composer-input textarea:focus,
#council-shell textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
.composer-note {
    color: rgba(244, 244, 245, 0.58);
    font-size: 12px;
    line-height: 1.2;
    text-align: center;
    margin: 6px 0 0;
}
.debate-switcher {
    width: 100%;
}
.debate-view-checkbox {
    position: fixed;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
}
.debate-view-checkbox:not(:checked) ~ .debate-final-view,
.debate-view-checkbox:not(:checked) ~ .debate-view-toggle .toggle-compare-label,
.debate-view-checkbox:checked ~ .debate-compare-view,
.debate-view-checkbox:checked ~ .debate-view-toggle .toggle-final-label {
    display: none !important;
}
.debate-view-toggle {
    position: fixed !important;
    left: calc(var(--history-sidebar-width) + 22px) !important;
    bottom: 92px !important;
    z-index: 80 !important;
    min-width: 88px !important;
    width: 88px !important;
    height: 42px !important;
    min-height: 42px !important;
    border-radius: 999px !important;
    color: #111111 !important;
    background: #f4f4f5 !important;
    border: 1px solid #f4f4f5 !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.45) !important;
}
.debate-view-toggle * {
    color: #111111 !important;
}
.debate-view-toggle:hover {
    background: #ffffff !important;
    border-color: #ffffff !important;
}

.options-drawer {
    width: min(100%, 1060px) !important;
    margin: 0 auto 10px !important;
    padding: 16px !important;
    background: #171717 !important;
    border: 1px solid #333333 !important;
    border-radius: 22px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55) !important;
}
.options-drawer,
.options-drawer * {
    color: var(--cai-text);
}
.drawer-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
}
.drawer-header h2 {
    color: var(--cai-text);
    font-size: 18px;
    line-height: 1.2;
    margin: 0;
}
.eyebrow,
.field-label {
    color: var(--cai-muted);
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.04em;
    margin: 0 0 8px;
    text-transform: uppercase;
}
.drawer-grid {
    gap: 10px !important;
    margin: 10px 0 0 !important;
}
.drawer-card,
.drawer-actions {
    background: #202020 !important;
    border: 1px solid #343434 !important;
    border-radius: 16px !important;
    padding: 12px !important;
}
.drawer-card .wrap,
.drawer-card .form,
.drawer-card fieldset,
.model-card .wrap,
.model-card .form {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.drawer-card label,
.model-card label {
    color: var(--cai-text) !important;
    font-size: 14px !important;
}
#council-shell input[type="radio"],
#council-shell input[type="checkbox"],
#council-shell input[type="range"] {
    accent-color: var(--cai-button) !important;
}
.drawer-actions {
    display: flex;
    flex-direction: column;
    gap: 8px;
    justify-content: center;
}
.drawer-actions button,
.drawer-actions .gr-button {
    width: 100% !important;
    min-height: 46px !important;
    color: var(--cai-text) !important;
    background: #2a2a2a !important;
    border: 1px solid #444444 !important;
    border-radius: 12px !important;
    font-weight: 650 !important;
}

.status-box {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.status-message {
    color: var(--cai-muted) !important;
    margin: 0 0 8px;
}
.connection-list {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}
.connection-list p,
.stat-card {
    margin: 0;
    padding: 10px 12px;
    background: #202020;
    border: 1px solid #343434;
    border-radius: 12px;
}
.connection-list p {
    display: flex;
    justify-content: space-between;
    gap: 10px;
}
.connection-list strong,
.connection-list span {
    color: var(--cai-text) !important;
    font-size: 13px;
}
.connection-list a {
    color: #8ab4ff !important;
    font-size: 13px;
}
.status-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
}
.stat-label {
    color: var(--cai-muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.04em;
    margin: 0 0 4px;
    text-transform: uppercase;
}
.stat-value {
    color: var(--cai-text);
    font-size: 13px;
    font-weight: 600;
    line-height: 1.35;
    margin: 0;
    overflow-wrap: anywhere;
}
.warning-list {
    background: var(--cai-warn-bg);
    border: 1px solid var(--cai-warn-line);
    border-radius: 12px;
    color: var(--cai-warn-text) !important;
    margin-top: 10px;
    padding: 10px 12px;
}
.warning-list,
.warning-list * {
    color: var(--cai-warn-text) !important;
}

@media (max-width: 760px) {
    :root {
        --cai-topbar-height: 96px;
        --cai-composer-height: 138px;
        --history-sidebar-width: 220px;
    }
    .answer-shell {
        padding: 20px 14px 66px !important;
    }
    .brand h1 {
        font-size: 22px;
    }
    .topbar {
        padding: 12px 14px !important;
    }
    .composer-shell {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .composer-bar {
        min-height: 56px;
        border-radius: 26px !important;
    }
    .execute-button,
    .execute-button button {
        min-width: 50px !important;
        font-size: 0 !important;
    }
    .execute-button::after {
        content: "Run";
        font-size: 14px;
    }
    .debate-view-toggle {
        left: calc(var(--history-sidebar-width) + 12px) !important;
        bottom: 142px !important;
        width: 76px !important;
        min-width: 76px !important;
        height: 38px !important;
        min-height: 38px !important;
    }
    .connection-list,
    .status-grid {
        grid-template-columns: 1fr;
    }
}
"""
