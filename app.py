from src.ui.gradio_app import build_app
from src.utils.config_loader import load_settings
from src.utils.edge_launcher import launch_controlled_edge


if __name__ == "__main__":
    settings = load_settings()
    demo, theme, custom_css = build_app(settings)
    host = settings["app"]["host"]
    port = settings["app"]["port"]
    app_url = f"http://{host}:{port}"
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
