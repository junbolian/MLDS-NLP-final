"""Three-tab Gradio app for the Multi-LexSum project."""

from __future__ import annotations

from pathlib import Path
import sys
import types

from app.inference import predict


EXAMPLE_LONG_CASE = """Plaintiffs challenge a city's practice of jailing people who cannot pay traffic-ticket debt.
They allege that the municipal court failed to assess ability to pay, converted fines to jail time,
and denied meaningful process before incarceration. The complaint seeks declaratory and injunctive relief,
arguing the practice violates due process, equal protection, and the prohibition on imprisoning indigent defendants
for nonpayment without an ability-to-pay inquiry."""

EXAMPLE_SHORT_CASE = """A group of detained immigrants alleges that officials denied access to medical care,
used prolonged detention without adequate review, and failed to provide meaningful hearings before transfer."""


def _import_gradio():
    """Import gradio while bypassing notebook-only IPython hooks when needed.

    In constrained environments, importing `gradio.ipython_ext` can trigger an
    IPython -> psutil path that fails before the web UI is ever needed. The
    app only needs Gradio's web components, so a tiny no-op extension stub is
    sufficient here.
    """
    if "gradio.ipython_ext" not in sys.modules:
        stub = types.ModuleType("gradio.ipython_ext")
        stub.load_ipython_extension = lambda *_args, **_kwargs: None
        sys.modules["gradio.ipython_ext"] = stub

    import gradio as gr
    if hasattr(gr, "utils") and hasattr(gr.utils, "colab_check"):
        gr.utils.colab_check = lambda: False

    return gr


def _format_prediction_block(payload: dict) -> str:
    lines = [
        f"Class action sought: {payload['class_action']['label']} ({payload['class_action']['confidence']:.3f})",
        f"Case type: {payload['case_type']['label']} ({payload['case_type']['confidence']:.3f})",
        "",
        "Models:",
        (
            f"- class_action: {payload['class_action']['model']} "
            f"[{payload['class_action']['model_kind']}, {payload['class_action']['text_source']}]"
        ),
        (
            f"- case_type: {payload['case_type']['model']} "
            f"[{payload['case_type']['model_kind']}, {payload['case_type']['text_source']}]"
        ),
    ]
    return "\n".join(lines)


def _format_metadata(metadata: dict) -> str:
    reduction = metadata["reduction"]
    lines = [
        f"Input chars: {metadata['input_text_chars']}",
        f"Reduction: {reduction['original_tokens']} -> {reduction['reduced_tokens']} tokens",
        f"Reduction method: {reduction['method']}",
        f"Selected sentences: {reduction['selected_sentences']}",
        "",
        "Warnings:",
        *[f"- {warning}" for warning in metadata["warnings"]],
    ]
    return "\n".join(lines)


def _run_app(case_text: str):
    result = predict(case_text)
    prediction_text = _format_prediction_block(result["predictions"])
    metadata_text = _format_metadata(result["metadata"])
    explainability = result["explainability"]
    return (
        result["summaries"]["long"],
        result["summaries"]["short"],
        result["summaries"]["tiny"],
        prediction_text,
        metadata_text,
        explainability["lr_class_action_plot"],
        explainability["lr_case_type_plot"],
        explainability["bert_attention"]["image_path"],
    )


def build_demo():
    try:
        gr = _import_gradio()
    except ImportError as exc:
        raise ImportError("Install gradio before launching the demo app.") from exc

    with gr.Blocks(title="Multi-LexSum Demo") as demo:
        gr.Markdown(
            """
            # Multi-LexSum Demo
            Paste a civil-rights case description or excerpt. The app will:
            1. generate long / short / tiny summaries
            2. classify whether a class action was sought
            3. predict the grouped case type
            4. render lightweight explanation artifacts
            """
        )
        with gr.Row():
            case_text = gr.Textbox(
                label="Case text",
                lines=16,
                placeholder="Paste case text here...",
                value=EXAMPLE_LONG_CASE,
            )
        with gr.Row():
            run_button = gr.Button("Run full pipeline", variant="primary")
            use_quick = gr.Button("Load quick example")
            use_long = gr.Button("Load demo example")

        with gr.Tabs():
            with gr.Tab("Summaries"):
                long_box = gr.Textbox(label="Long summary", lines=10)
                short_box = gr.Textbox(label="Short summary", lines=6)
                tiny_box = gr.Textbox(label="Tiny summary", lines=2)
            with gr.Tab("Predictions"):
                prediction_box = gr.Textbox(label="Predictions", lines=8)
                metadata_box = gr.Textbox(label="Run metadata", lines=8)
            with gr.Tab("Explainability"):
                gr.Markdown(
                    "LR SHAP charts are live artifacts for the generated long summary. "
                    "The BERT attention heatmap is illustrative, not causal proof."
                )
                lr_class_action_image = gr.Image(label="LR SHAP — class action", type="filepath")
                lr_case_type_image = gr.Image(label="LR SHAP — case type", type="filepath")
                bert_attention_image = gr.Image(label="BERT attention heatmap", type="filepath")

        run_button.click(
            _run_app,
            inputs=[case_text],
            outputs=[
                long_box,
                short_box,
                tiny_box,
                prediction_box,
                metadata_box,
                lr_class_action_image,
                lr_case_type_image,
                bert_attention_image,
            ],
        )
        use_quick.click(lambda: EXAMPLE_SHORT_CASE, outputs=[case_text])
        use_long.click(lambda: EXAMPLE_LONG_CASE, outputs=[case_text])
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
