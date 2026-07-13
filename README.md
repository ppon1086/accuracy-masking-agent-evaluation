# Accuracy Masking in LLM-as-Judge Evaluation

Code to reproduce all tables and figures from:

> **Selecting LLM Judges for Agent Evaluation Pipelines: Accuracy Masking and Review-Burden Tradeoffs**

## Data

Download the judgment files and annotations from [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench):

```bash
pip install huggingface_hub

python - << 'PYTHON'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="McGill-NLP/agent-reward-bench",
    repo_type="dataset",
    local_dir="./agent-reward-bench",
    allow_patterns=[
        # GPT-4o agent — all benchmarks, all 5 judges
        "judgments/*/GenericAgent-gpt-4o-2024-11-20/**/*.json",
        # Claude agent — all benchmarks, all 5 judges
        "judgments/*/GenericAgent-anthropic_claude-3.7-sonnet/**/*.json",
        # Qwen agent — all benchmarks, all 5 judges
        "Qwen/**/*.json",
        # Expert annotations
        "annotations.csv",
    ],
)
PYTHON
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python accuracy_masking_analysis.py \
    --gpt4o_zip   ./agent-reward-bench/judgments_gpt4o.zip \
    --claude_zip  ./agent-reward-bench/judgments_claude.zip \
    --qwen_zip    ./agent-reward-bench/Qwen.zip \
    --annotations ./agent-reward-bench/annotations.csv \
    --output_dir  ./output
```

This reproduces:
- **Table 3**: Accuracy ranges across judges per evaluated agent
- **Table 4**: Side-effect failure profiles (FNR/FPR)
- **Figure 1**: Side-effect recall vs. review burden (three-panel scatter)
- **Appendix Table 6**: Full per-judge accuracy
- **Appendix Table 7**: Full SE FNR/FPR per agent
- **Appendix Table 8**: Looping accuracy by label slice
- **Appendix Tables 9/10**: Logistic regression coefficients

## Citation

```bibtex
@inproceedings{yourkey2025accuracy,
  title     = {Selecting {LLM} Judges for Agent Evaluation Pipelines:
               Accuracy Masking and Review-Burden Tradeoffs},
  author    = {Your Name},
  booktitle = {Proceedings of ...},
  year      = {2025},
}
```
