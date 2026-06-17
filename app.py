from src.ui.gradio_app import build_app
from src.utils.config_loader import load_settings
from src.utils.edge_launcher import launch_controlled_edge


APP_VERSION_LABEL = "CouncilAI v1"


if __name__ == "__main__":
    settings = load_settings()
    demo, theme, custom_css = build_app(settings)
    host = settings["app"]["host"]
    port = settings["app"]["port"]
    app_url = f"http://{host}:{port}"
    print(f"{APP_VERSION_LABEL}\n", flush=True)
    demo.launch(
        server_name=host,
        server_port=port,
        prevent_thread_lock=True,
        theme=theme,
        css=custom_css,
    )
    result = launch_controlled_edge(settings, app_url)
    print(result.message)
    demo.block_thread()
