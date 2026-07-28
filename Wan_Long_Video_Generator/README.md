# Wan Long Video Generator

This Gradio app wraps the Wan `generate.py` command so text-to-video generation can be launched from a browser.

## Usage

Install Gradio in the same Python environment used by Wan:

```bash
pip install gradio
```

Run the UI from the Wan2.1 repository root, where `generate.py` is available:

```bash
python Wan_Long_Video_Generator/app.py
```

Open the printed Gradio URL and click **Generate Video**. The default form values match this command:

```bash
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
  --use_prompt_extend \
  --prompt_extend_method local_qwen \
  --prompt_extend_target_lang ch
```

If `generate.py` or outputs are somewhere else, change the **Advanced** fields or set environment variables:

```bash
GENERATE_PY=/path/to/generate.py WAN_OUTPUT_DIR=/path/to/output python Wan_Long_Video_Generator/app.py
```
