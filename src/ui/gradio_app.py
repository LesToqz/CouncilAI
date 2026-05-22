from __future__ import annotations

import html
from typing import Any

import gradio as gr

from src.browser.edge_manager import EdgeManager
from src.orchestration.graph import DebateGraph
from src.orchestration.state import DebateState
from src.utils.edge_launcher import launch_controlled_edge
from src.utils.text_cleaner import clean_response_text


MODEL_ORDER = ("chatgpt", "gemini", "claude")


def build_app(settings: dict) -> tuple[gr.Blocks, Any, str]:
    app_name = settings.get("app", {}).get("name", "CouncilAI")
    max_iterations = int(settings.get("debate", {}).get("max_iterations", 5))
    default_iterations = int(settings.get("debate", {}).get("default_iterations", 2))

    async def handle_check_tabs() -> str:
        return await check_existing_tabs(settings)

    async def handle_run(
        prompt: str,
        mode: str,
        iterations: float,
        use_chatgpt: bool,
        use_gemini: bool,
        use_claude: bool,
    ) -> tuple[str, Any, str]:
        return await run_debate(
            settings,
            prompt,
            mode,
            iterations,
            use_chatgpt,
            use_gemini,
            use_claude,
        )

    def toggle_options(is_open: bool) -> tuple[bool, Any, Any]:
        next_open = not bool(is_open)
        return (
            next_open,
            gr.update(visible=next_open),
            gr.update(value="x" if next_open else "+"),
        )

    theme = gr.themes.Soft(
        primary_hue="gray",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    )

    with gr.Blocks(title=app_name, elem_id="council-shell") as demo:
        drawer_open = gr.State(False)

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
            final_answer_box = gr.Markdown(
                value=_welcome_markdown(),
                label="Final Answer",
                elem_classes=["final-answer-box"],
            )
            observable_box = gr.Markdown(
                visible=False,
                label="Observable Debate",
                elem_classes=["debate-trace-box"],
                sanitize_html=False,
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
                            choices=["Silent Final Mode", "Observable Debate Mode"],
                            value="Silent Final Mode",
                            label="Mode",
                            show_label=False,
                            container=False,
                        )
                    with gr.Column(scale=7, elem_classes=["drawer-card"]):
                        gr.HTML('<p class="field-label">Iterations</p>')
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

        check_tabs_button.click(
            fn=handle_check_tabs,
            inputs=None,
            outputs=status_box,
        )

        mode_radio.change(
            fn=lambda mode: gr.update(visible=mode == "Observable Debate Mode"),
            inputs=mode_radio,
            outputs=observable_box,
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
            outputs=[status_box, observable_box, final_answer_box],
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
            outputs=[status_box, observable_box, final_answer_box],
        )

    return demo, theme, _custom_css()


async def check_existing_tabs(settings: dict) -> str:
    app_settings = settings.get("app", {})
    app_url = f"http://{app_settings.get('host', '127.0.0.1')}:{app_settings.get('port', 7860)}"
    launch_result = launch_controlled_edge(settings, app_url)
    manager = EdgeManager(settings)
    statuses = [f"<p class=\"status-message\">{html.escape(launch_result.message)}</p>", '<div class="connection-list">']
    try:
        found = await manager.find_existing_ai_tabs()
        for model_key in MODEL_ORDER:
            site = settings["model_sites"][model_key]
            page_url = found.get(model_key)
            if page_url:
                statuses.append(f"<p><strong>{html.escape(site['name'])}</strong><span>Connected</span></p>")
            else:
                site_url = site["url"]
                statuses.append(
                    f'<p><strong>{html.escape(site["name"])}</strong>'
                    f'<a href="{html.escape(site_url)}" target="_blank">Open tab</a></p>'
                )
    except Exception as exc:  # noqa: BLE001
        statuses.append(f"<p><strong>Attach failed</strong><span>{html.escape(str(exc))}</span></p>")
    finally:
        await manager.stop()

    statuses.append("</div>")
    return "\n".join(statuses)


async def run_debate(
    settings: dict,
    prompt: str,
    mode_label: str,
    iterations: float,
    use_chatgpt: bool,
    use_gemini: bool,
    use_claude: bool,
) -> tuple[str, Any, str]:
    prompt = (prompt or "").strip()
    active_models = _selected_models(use_chatgpt, use_gemini, use_claude)
    mode = "observable" if mode_label == "Observable Debate Mode" else "silent"

    if not prompt:
        return _message_panel("Prompt required", "Enter a prompt before running a debate."), gr.update(value="", visible=mode == "observable"), _welcome_markdown()
    if len(active_models) < 2:
        return _message_panel("Model selection", "Select at least two models."), gr.update(value="", visible=mode == "observable"), _welcome_markdown()

    max_iterations = int(settings.get("debate", {}).get("max_iterations", 5))
    iteration_count = max(1, min(int(iterations), max_iterations))

    state = DebateState(
        user_prompt=prompt,
        mode=mode,
        max_iterations=iteration_count,
        active_models=active_models,
    )

    graph = DebateGraph(settings)
    try:
        final_state = await graph.ainvoke(state)
    except Exception as exc:  # noqa: BLE001 - UI must show unexpected orchestration failures.
        return (
            _message_panel("Debate failed", str(exc)),
            gr.update(value="", visible=mode == "observable"),
            "",
        )

    status = _status_markdown(final_state)
    observable = _observable_markdown(final_state) if mode == "observable" else ""
    return status, gr.update(value=observable, visible=mode == "observable"), _format_final_answer(final_state.final_answer)


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


def _observable_markdown(state: DebateState) -> str:
    sections = ["## Debate Trace"]
    for turn in state.turns:
        title = f"{turn.model.upper()} | iteration {turn.iteration} | {turn.phase.replace('_', ' ')}"
        sections.append(f"<details><summary>{html.escape(title)}</summary>\n\n")
        if turn.error:
            sections.append(f"**Error:** {html.escape(turn.error)}\n\n")
        if turn.response:
            sections.append(turn.response)
        sections.append("\n\n</details>")
    return "\n".join(sections)


def _format_final_answer(answer: str | None) -> str:
    return clean_response_text(answer or "")


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
}

.topbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 70;
    width: 100% !important;
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
    left: 0;
    right: 0;
    top: var(--cai-topbar-height);
    bottom: var(--cai-composer-height);
    z-index: 1;
    width: 100% !important;
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
    left: 0;
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
.final-answer-box {
    max-width: 960px;
    margin: 0 auto !important;
    padding: 0 0 20px !important;
}
.debate-trace-box {
    max-width: 960px;
    margin: 22px auto 0 !important;
    padding: 20px 0 0 !important;
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

.composer-shell {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 60;
    display: flex;
    align-items: center;
    width: 100% !important;
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
    .connection-list,
    .status-grid {
        grid-template-columns: 1fr;
    }
}
"""
