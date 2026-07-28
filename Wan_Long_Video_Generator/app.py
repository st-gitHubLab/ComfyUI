"""Gradio UI for Wan text-to-video generation.

This wrapper exposes the command-line workflow below as a web UI::

    python generate.py --task t2v-1.3B --size 832*480 \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
        --use_prompt_extend --prompt_extend_method local_qwen \
        --prompt_extend_target_lang ch

Run from the Wan2.1 repository root, or set GENERATE_PY to the target
``generate.py`` path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import gradio as gr

DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage"
)
DEFAULT_TASK = "t2v-1.3B"
DEFAULT_SIZE = "832*480"
DEFAULT_CKPT_DIR = "./Wan2.1-T2V-1.3B"
DEFAULT_PROMPT_EXTEND_METHOD = "local_qwen"
DEFAULT_PROMPT_EXTEND_TARGET_LANG = "ch"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _resolve_generate_py(generate_py: str) -> Path:
    path = Path(generate_py).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"generate.py not found: {path}")
    return path


def _snapshot_video_files(search_dir: Path) -> dict[Path, float]:
    if not search_dir.exists():
        return {}
    return {
        path: path.stat().st_mtime
        for path in search_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def _find_newest_video(search_dir: Path, before: dict[Path, float], started_at: float) -> str | None:
    candidates: list[Path] = []
    for path in search_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        mtime = path.stat().st_mtime
        if path not in before or mtime > before[path] or mtime >= started_at:
            candidates.append(path)
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item.stat().st_mtime))


def build_command(
    generate_py: str,
    task: str,
    size: str,
    ckpt_dir: str,
    prompt: str,
    use_prompt_extend: bool,
    prompt_extend_method: str,
    prompt_extend_target_lang: str,
) -> list[str]:
    script = _resolve_generate_py(generate_py)
    command = [
        sys.executable,
        str(script),
        "--task",
        task,
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--prompt",
        prompt,
    ]
    if use_prompt_extend:
        command.extend(
            [
                "--use_prompt_extend",
                "--prompt_extend_method",
                prompt_extend_method,
                "--prompt_extend_target_lang",
                prompt_extend_target_lang,
            ]
        )
    return command


def generate_video(
    prompt: str,
    task: str,
    size: str,
    ckpt_dir: str,
    use_prompt_extend: bool,
    prompt_extend_method: str,
    prompt_extend_target_lang: str,
    generate_py: str,
    output_search_dir: str,
) -> Iterator[tuple[str | None, str]]:
    """Run Wan generate.py and stream terminal output into Gradio."""
    if not prompt.strip():
        yield None, "Prompt cannot be empty."
        return

    search_dir = Path(output_search_dir or ".").expanduser()
    if not search_dir.is_absolute():
        search_dir = Path.cwd() / search_dir

    started_at = time.time()
    before = _snapshot_video_files(search_dir)

    try:
        command = build_command(
            generate_py,
            task,
            size,
            ckpt_dir,
            prompt,
            use_prompt_extend,
            prompt_extend_method,
            prompt_extend_target_lang,
        )
    except Exception as exc:  # noqa: BLE001 - show UI-friendly setup errors
        yield None, f"Failed to build command: {exc}"
        return

    log_lines = ["Running command:", " ".join(command), ""]
    yield None, "\n".join(log_lines)

    process = subprocess.Popen(
        command,
        cwd=str(Path(generate_py).expanduser().resolve().parent if Path(generate_py).expanduser().is_absolute() else Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line.rstrip())
        yield None, "\n".join(log_lines[-300:])

    return_code = process.wait()
    video_path = _find_newest_video(search_dir, before, started_at)
    if return_code != 0:
        log_lines.append(f"Generation failed with exit code {return_code}.")
        yield video_path, "\n".join(log_lines[-300:])
        return

    log_lines.append("Generation finished successfully.")
    if video_path:
        log_lines.append(f"Detected output video: {video_path}")
    else:
        log_lines.append("No new video file was detected. Check the generate.py output path in the log.")
    yield video_path, "\n".join(log_lines[-300:])


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Wan Long Video Generator") as demo:
        gr.Markdown("# Wan Long Video Generator\nUse Gradio to call `generate.py` for Wan text-to-video generation.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
                with gr.Row():
                    task = gr.Textbox(label="Task", value=DEFAULT_TASK)
                    size = gr.Textbox(label="Size", value=DEFAULT_SIZE)
                ckpt_dir = gr.Textbox(label="Checkpoint directory", value=DEFAULT_CKPT_DIR)
                use_prompt_extend = gr.Checkbox(label="Use prompt extension", value=True)
                with gr.Row():
                    prompt_extend_method = gr.Textbox(label="Prompt extension method", value=DEFAULT_PROMPT_EXTEND_METHOD)
                    prompt_extend_target_lang = gr.Textbox(label="Prompt extension target language", value=DEFAULT_PROMPT_EXTEND_TARGET_LANG)
                with gr.Accordion("Advanced", open=False):
                    generate_py = gr.Textbox(label="generate.py path", value=os.environ.get("GENERATE_PY", "generate.py"))
                    output_search_dir = gr.Textbox(label="Output search directory", value=os.environ.get("WAN_OUTPUT_DIR", "."))
                run_button = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=1):
                video = gr.Video(label="Generated video")
                logs = gr.Textbox(label="Logs", lines=22, max_lines=30)

        run_button.click(
            generate_video,
            inputs=[prompt, task, size, ckpt_dir, use_prompt_extend, prompt_extend_method, prompt_extend_target_lang, generate_py, output_search_dir],
            outputs=[video, logs],
        )
    return demo


if __name__ == "__main__":
    create_ui().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
