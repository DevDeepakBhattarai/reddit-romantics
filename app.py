"""Convenience entry point: `python app.py` launches the Gradio UI."""

from reddit_video.ui import build_ui

if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
