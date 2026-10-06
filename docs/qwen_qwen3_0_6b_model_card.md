# Qwen/Qwen3-0.6B

**Model Type:** Text Generation / Causal Language Model  
**License:** Apache 2.0  
**Tags:** `qwen3`, `transformers`, `pytorch`, `safetensors`, `text-generation-inference`, `vllm`

---

## Table of Contents

- [Qwen3 Highlights](#qwen3-highlights)
- [Model Overview](#model-overview)
- [Quickstart](#quickstart)
- [Switching Between Thinking and Non-Thinking Mode](#switching-between-thinking-and-non-thinking-mode)
- [Agentic Use](#agentic-use)
- [Best Practices](#best-practices)
- [Citation](#citation)

---

## Qwen3 Highlights

Qwen3 is the latest generation of large language models in the Qwen series, offering a comprehensive suite of dense and Mixture-of-Experts (MoE) models. Key feature highlights:

- **Multilingual Capabilities:** Support for 100+ languages and dialects with strong performance in multilingual instruction following and translation.
- **Dual Thinking Modes:** Supports both reasoning ("thinking") mode for complex problem-solving and standard ("non-thinking") mode for fast responses.
- **Agentic Capabilities:** High performance on tool use, function calling, and complex agentic workflows.

---

## Model Overview

`Qwen3-0.6B` has the following key features and specifications:

| Feature | Specification |
| :--- | :--- |
| **Model Type** | Causal Language Model |
| **Training Stage** | Pretraining & Post-training |
| **Total Parameters** | 0.6B |
| **Non-Embedding Parameters** | 0.44B |
| **Number of Layers** | 28 |
| **Attention Heads (GQA)** | 16 for Q, 8 for KV |
| **Context Length** | 32,768 tokens |

For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to the official [Qwen GitHub Repository](https://github.com/QwenLM/Qwen).

---

## Quickstart

### Using `transformers`

> **Note:** Ensure you are using `transformers >= 4.51.0`. Older versions will raise a `KeyError: 'qwen3'`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to Large Language Models."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Set to False for non-thinking mode
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Server Frameworks

#### vLLM
```bash
vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
```

#### SGLang
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3
```

---

## Switching Between Thinking and Non-Thinking Mode

### Chat Template Flag

You can toggle reasoning capabilities directly via the `enable_thinking` argument in `apply_chat_template`:

```python
# Enable Reasoning (Thinking Mode)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

# Disable Reasoning (Non-Thinking Mode)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

### Dynamic Switch via User Input

When `enable_thinking=True`, users can dynamically override thinking behavior using tags directly in their prompt:
- Use `/think` to force reasoning generation.
- Use `/no_think` to bypass reasoning generation.

---

## Agentic Use

Qwen3 excels in tool-calling and function-calling tasks. We recommend using **Qwen-Agent** for building complex workflows:

```bash
pip install qwen-agent
```

Qwen-Agent handles formatting, parsing, and execution loop orchestration internally for optimal agent reliability.

---

## Best Practices

### Recommended Sampling Parameters

- **Thinking Mode (`enable_thinking=True`):**
  - `Temperature`: `0.6`
  - `TopP`: `0.95`
  - `TopK`: `20`
  - `MinP`: `0.0`
  - *Do NOT use greedy decoding*, as it may lead to performance degradation or infinite reasoning loops.

- **Non-Thinking Mode (`enable_thinking=False`):**
  - `Temperature`: `0.7`
  - `TopP`: `0.8`
  - `TopK`: `20`
  - `MinP`: `0.0`

### Repetition Prevention
If you observe endless repetitions, set `presence_penalty` between `0` and `2.0` (recommended: `1.5` for quantized variants).

### Output Formatting Guidelines
- **Math Problems:** Include `"Please reason step by step, and put your final answer within \\boxed{}."` in your prompt.
- **Multiple-Choice Questions:** Standardize output by requesting JSON format: `"Please show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\"."`

---

## Citation

If you find our work helpful, please cite the technical report:

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
```