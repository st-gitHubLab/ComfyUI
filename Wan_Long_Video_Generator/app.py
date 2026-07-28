"""Gradio entry point for Wan Long Video Generator."""
from __future__ import annotations

import gradio as gr

from core.pipeline import LongVideoPipeline


def generate_video(prompt: str, total_seconds: int) -> str:
    """Gradio callback that returns the generated video path."""
    pipeline = LongVideoPipeline()
    return str(pipeline.generate(prompt, total_seconds))


with gr.Blocks(title="Wan Long Video Generator") as demo:
    gr.Markdown("# Wan Long Video Generator\n使用 Wan2.1 T2V + I2V 分段续接生成长视频。")
    prompt = gr.Textbox(label="提示词", lines=4, placeholder="输入长视频描述")
    total_seconds = gr.Slider(label="总时长（秒）", minimum=5, maximum=120, value=20, step=5)
    run_button = gr.Button("生成视频")
    output = gr.Video(label="最终视频")
    run_button.click(generate_video, inputs=[prompt, total_seconds], outputs=output)


if __name__ == "__main__":
    demo.launch()
