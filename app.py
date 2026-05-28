"""Hugging Face Spaces entrypoint."""

from app.gradio_app import build_demo

demo = build_demo()


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
