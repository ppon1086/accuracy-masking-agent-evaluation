"""
Accuracy Masking in LLM-as-Judge Evaluation
============================================
Reproduces all tables and figures from:
  "Selecting LLM Judges for Agent Evaluation Pipelines:
   Accuracy Masking and Review-Burden Tradeoffs"

Usage:
    python accuracy_masking_analysis.py \
        --gpt4o_zip   path/to/gpt4o_judgments.zip \
        --claude_zip  path/to/claude_judgments.zip \
        --qwen_zip    path/to/qwen_judgments.zip \
        --annotations path/to/annotations.csv \
        --output_dir  ./output

Data:
    Download from AgentRewardBench:
    https://huggingface.co/datasets/McGill-NLP/agent-reward-bench

    annotations.csv is included in the AgentRewardBench repository.

    For each agent zip, use the judgment files available at:
    - GPT-4o agent:    judgments/ folder (bench/agent/judge/file structure)
    - Claude agent:    judgments_claude/ folder
    - Qwen agent:      Qwen/ folder
"""

import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression

# ── Constants ────────────────────────────────────────────────────────────────

JUDGE_MAP = {
    "gpt-4o-mini-noaxtree":       "GPT-4o-mini",
    "gpt-4o-noaxtree":            "GPT-4o",
    "claude-3.7-sonnet-noaxtree": "Claude-3.7-Sonnet",
    "llama-3.3-70b-noscreen":     "Llama-3.3-70B",
    "qwen-2.5-vl-noaxtree":       "Qwen-2.5-VL",
}

JUDGES = [
    "GPT-4o-mini",
    "GPT-4o",
    "Claude-3.7-Sonnet",
    "Llama-3.3-70B",
    "Qwen-2.5-VL",
]

AGENT_MODELS = {
    "gpt4o":  "GenericAgent-gpt-4o-2024-11-20",
    "claude": "GenericAgent-anthropic_claude-3.7-sonnet",
    "qwen":   "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct",
}

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_judge_response(content):
    """Extract structured judge output from response text."""
    def extract_tag(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    def to_bin(val, positive_values):
        if val is None:
            return None
        return 1 if val.lower().strip() in positive_values else 0

    return {
        "judge_success":     to_bin(extract_tag("success"), ("successful",)),
        "judge_side_effect": to_bin(extract_tag("side"),    ("yes", "true")),
        "judge_looping":     to_bin(extract_tag("loop"),    ("yes", "true")),
    }


def load_judgments(zip_path, bench_idx, judge_idx):
    """
    Load judgment files from a zip archive.

    Args:
        zip_path:   Path to zip file
        bench_idx:  Index of benchmark name in split path
        judge_idx:  Index of judge folder name in split path
    """
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        for fname in [f for f in zf.namelist() if f.endswith(".json")]:
            parts = fname.split("/")
            bench = parts[bench_idx]
            judge_folder = parts[judge_idx]
            jname = JUDGE_MAP.get(judge_folder)
            if not jname:
                continue
            task_id = parts[-1].replace(".json", "")
            with zf.open(fname) as f:
                d = json.load(f)
            parsed = parse_judge_response(
                d["response"]["choices"][0]["message"]["content"]
            )
            summary = d.get("trajectory_info", {}).get("summary_info", {})
            records.append({
                "traj_key":    f"{bench}__{task_id}",
                "task_id":     task_id,
                "benchmark":   bench,
                "judge_model": jname,
                "n_steps":     summary.get("n_steps"),
                **parsed,
            })
    df = pd.DataFrame(records).drop_duplicates(subset=["traj_key", "judge_model"])
    print(f"  Loaded {df['traj_key'].nunique()} trajectories × "
          f"{df['judge_model'].nunique()} judges = {len(df)} rows")
    return df


# ── Expert labels ─────────────────────────────────────────────────────────────

def load_expert_labels(annotations_path, model_name):
    """
    Load and aggregate expert annotations for one agent model.
    Applies majority vote; ties and unsure labels are excluded per dimension.
    """
    raw = pd.read_csv(annotations_path)
    agent = raw[raw["model_name"] == model_name].copy()

    def parse_success(v):
        if pd.isna(v): return None
        v = str(v).strip()
        if v == "Successful":   return 1
        if v == "Unsuccessful": return 0
        return None  # Unsure -> excluded

    def parse_yn(v):
        if pd.isna(v): return None
        v = str(v).strip()
        return 1 if v == "Yes" else (0 if v == "No" else None)

    agent["success_bin"]     = agent["trajectory_success"].apply(parse_success)
    agent["side_effect_bin"] = agent["trajectory_side_effect"].apply(parse_yn)
    agent["looping_bin"]     = agent["trajectory_looping"].apply(parse_yn)

    def majority(series):
        vals = series.dropna()
        if len(vals) == 0: return None
        s = vals.sum(); n = len(vals)
        if s > n / 2: return 1
        if s < n / 2: return 0
        return None  # tie -> excluded

    agg = agent.groupby(["benchmark", "task_id"]).agg(
        success_bin=("success_bin", majority),
        side_effect_bin=("side_effect_bin", majority),
        looping_bin=("looping_bin", majority),
    ).reset_index()
    agg["traj_key"] = agg["benchmark"] + "__" + agg["task_id"].astype(str)

    se_pos   = int(agg["side_effect_bin"].sum())
    loop_pos = int(agg["looping_bin"].sum())
    print(f"  Expert: {len(agg)} trajectories, "
          f"SE_pos={se_pos}, Loop_pos={loop_pos}")
    return agg


def merge_with_expert(judgments, expert):
    """Merge judgments with expert labels; add per-dimension error columns."""
    m = judgments.merge(
        expert[["traj_key", "success_bin", "side_effect_bin", "looping_bin"]],
        on="traj_key", how="inner"
    )
    for jcol, ecol, errcol in [
        ("judge_success",     "success_bin",     "error_success"),
        ("judge_side_effect", "side_effect_bin", "error_side_effect"),
        ("judge_looping",     "looping_bin",     "error_looping"),
    ]:
        m[errcol] = np.where(
            m[ecol].isna(),
            np.nan,
            (m[jcol] != m[ecol]).astype(float)
        )
    print(f"  Merged: {m['traj_key'].nunique()} trajectories × "
          f"{m['judge_model'].nunique()} judges = {len(m)} rows")
    return m


# ── Metrics ───────────────────────────────────────────────────────────────────

def dimension_accuracy(merged, judge, dim_col, expert_col):
    """Accuracy on a dimension, excluding NaN expert labels."""
    s = merged[merged["judge_model"] == judge]
    valid = s.dropna(subset=[expert_col])
    if len(valid) == 0:
        return np.nan
    return (valid[dim_col] == valid[expert_col]).mean()


def se_fnr_fpr(merged, judge):
    """False-negative and false-positive rates on side-effect dimension."""
    s = merged[merged["judge_model"] == judge]
    pos = s[s["side_effect_bin"] == 1]["error_side_effect"].dropna()
    neg = s[s["side_effect_bin"] == 0]["error_side_effect"].dropna()
    fnr = pos.mean() if len(pos) > 0 else np.nan
    fpr = neg.mean() if len(neg) > 0 else np.nan
    return fnr, fpr, len(pos)


def bootstrap_ci(series, n_boot=1000, ci=95):
    """Bootstrap confidence interval for the mean."""
    series = series.dropna()
    if len(series) < 3:
        return np.nan, np.nan
    boot = [series.sample(len(series), replace=True).mean()
            for _ in range(n_boot)]
    lo = (100 - ci) / 2
    return np.percentile(boot, lo), np.percentile(boot, 100 - lo)


def fisher_p(pos, neg):
    """Fisher's exact test p-value comparing error rates."""
    if len(pos) < 3:
        return np.nan
    table = [
        [int(pos.sum()), int(len(pos) - pos.sum())],
        [int(neg.sum()), int(len(neg) - neg.sum())],
    ]
    _, p = fisher_exact(table)
    return p


def sig_stars(p):
    if np.isnan(p): return "n.s."
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "n.s."


# ── Table generators ──────────────────────────────────────────────────────────

def table_accuracy_ranges(agent_data):
    """Table 3: accuracy ranges across judges per agent."""
    print("\n" + "=" * 65)
    print("TABLE 3 — ACCURACY RANGES ACROSS JUDGES")
    print("=" * 65)
    print(f"{'Agent':<20} {'Success range':>14} {'SE range':>10} {'Loop range':>11}")
    print("-" * 57)
    for agent_name, merged in agent_data.items():
        succs, ses, loops = [], [], []
        for j in JUDGES:
            succs.append(dimension_accuracy(merged, j, "judge_success",     "success_bin"))
            ses.append(  dimension_accuracy(merged, j, "judge_side_effect", "side_effect_bin"))
            loops.append(dimension_accuracy(merged, j, "judge_looping",     "looping_bin"))
        sr  = (max(succs) - min(succs)) * 100
        ser = (max(ses)   - min(ses))   * 100
        lr  = (max(loops) - min(loops)) * 100
        print(f"  {agent_name:<18} {sr:>13.1f}pp {ser:>9.1f}pp {lr:>10.1f}pp")


def table_se_profiles(agent_data):
    """Table 4: SE FNR/FPR failure profiles."""
    print("\n" + "=" * 75)
    print("TABLE 4 — SIDE-EFFECT FAILURE PROFILES")
    print("=" * 75)
    for agent_name, merged in agent_data.items():
        n_se = int(merged.drop_duplicates("traj_key")["side_effect_bin"].sum())
        print(f"\n  {agent_name} agent (N(SE=1)={n_se}):")
        print(f"  {'Judge':<22} {'Recall':>7} {'FNR':>7} {'FPR':>7} {'p':>8}  Profile")
        print("  " + "-" * 65)
        for j in JUDGES:
            fnr, fpr, n_pos = se_fnr_fpr(merged, j)
            recall = 1 - fnr
            s = merged[merged["judge_model"] == j]
            pos = s[s["side_effect_bin"] == 1]["error_side_effect"].dropna()
            neg = s[s["side_effect_bin"] == 0]["error_side_effect"].dropna()
            p = fisher_p(pos, neg)
            stars = sig_stars(p)
            if fnr > 0.5:   profile = "Silent misser"
            elif fpr > 0.5: profile = "Over-flagger"
            else:           profile = "Balanced/Inconclusive"
            print(f"  {j:<22} {recall:>7.3f} {fnr:>7.3f} {fpr:>7.3f} "
                  f"{stars:>8}  {profile}")


def table_appendix_accuracy(agent_data):
    """Appendix Table 6: full per-judge accuracy."""
    print("\n" + "=" * 65)
    print("APPENDIX TABLE 6 — FULL PER-JUDGE ACCURACY")
    print("=" * 65)
    print(f"{'Agent':<10} {'Judge':<24} {'Success':>8} {'SE':>8} {'Loop':>8}")
    print("-" * 60)
    for agent_name, merged in agent_data.items():
        for j in JUDGES:
            sa  = dimension_accuracy(merged, j, "judge_success",     "success_bin")
            sea = dimension_accuracy(merged, j, "judge_side_effect", "side_effect_bin")
            la  = dimension_accuracy(merged, j, "judge_looping",     "looping_bin")
            print(f"  {agent_name:<8} {j:<24} {sa:>8.3f} {sea:>8.3f} {la:>8.3f}")
        print()


def table_appendix_se(agent_data):
    """Appendix Table 7: full SE FNR/FPR per agent."""
    print("\n" + "=" * 75)
    print("APPENDIX TABLE 7 — FULL SE FNR/FPR")
    print("=" * 75)
    for agent_name, merged in agent_data.items():
        n_se = int(merged.drop_duplicates("traj_key")["side_effect_bin"].sum())
        print(f"\n  {agent_name} (N(SE=1)={n_se}):")
        print(f"  {'Judge':<24} {'Recall':>8} {'FNR':>8} {'FPR':>8} {'p':>8}")
        print("  " + "-" * 55)
        for j in JUDGES:
            s   = merged[merged["judge_model"] == j]
            pos = s[s["side_effect_bin"] == 1]["error_side_effect"].dropna()
            neg = s[s["side_effect_bin"] == 0]["error_side_effect"].dropna()
            fnr = pos.mean(); fpr = neg.mean(); recall = 1 - fnr
            p = fisher_p(pos, neg)
            r_lo, r_hi = bootstrap_ci(1 - pos)
            print(f"  {j:<24} {recall:>8.3f} {fnr:>8.3f} {fpr:>8.3f} "
                  f"{sig_stars(p):>8}  "
                  f"Recall CI=[{r_lo:.2f},{r_hi:.2f}]")


def table_appendix_looping(agent_data):
    """Appendix Table 8: looping by label slice."""
    print("\n" + "=" * 75)
    print("APPENDIX TABLE 8 — LOOPING BY LABEL SLICE")
    print("=" * 75)
    for agent_name, merged in agent_data.items():
        n_loop = int(merged.drop_duplicates("traj_key")["looping_bin"].sum())
        print(f"\n  {agent_name} (N(Loop=1)={n_loop}):")
        print(f"  {'Judge':<24} {'Overall':>8} {'Loop=YES':>9} "
              f"{'Loop=NO':>8} {'Delta':>8} {'p':>8}")
        print("  " + "-" * 68)
        for j in JUDGES:
            s       = merged[merged["judge_model"] == j]
            overall = dimension_accuracy(merged, j, "judge_looping", "looping_bin")
            lo_valid = s.dropna(subset=["looping_bin"])
            yes = lo_valid[lo_valid["looping_bin"] == 1]["error_looping"].dropna()
            no  = lo_valid[lo_valid["looping_bin"] == 0]["error_looping"].dropna()
            ay  = 1 - yes.mean()
            an  = 1 - no.mean()
            delta = (ay - overall) * 100
            p = fisher_p(yes, no)
            print(f"  {j:<24} {overall:>8.3f} {ay:>9.3f} {an:>8.3f} "
                  f"{delta:>+8.1f}pp {sig_stars(p):>8}")


# ── Regression ────────────────────────────────────────────────────────────────

def run_regression(gpt4o_merged):
    """
    Appendix Tables 9/10: logistic regression on GPT-4o agent trajectories.
    Uses trajectory-clustered bootstrap for CIs (1000 iterations).
    Reference: Claude-3.7-Sonnet on WebArena.
    """
    print("\n" + "=" * 65)
    print("APPENDIX TABLES 9/10 — LOGISTIC REGRESSION")
    print("Reference: Claude-3.7-Sonnet on WebArena")
    print("=" * 65)

    df = gpt4o_merged.copy()

    # Long trajectory = top tertile by step count
    steps = df.drop_duplicates("traj_key").set_index("traj_key")["n_steps"]
    threshold = steps.quantile(2 / 3)
    long_trajs = set(steps[steps >= threshold].index)
    df["long_traj"] = df["traj_key"].isin(long_trajs).astype(int)
    print(f"\n  Long trajectory threshold: {threshold:.0f} steps")
    print(f"  Long trajectories: {len(long_trajs)}")

    # Dummies (reference: Claude-3.7-Sonnet, WebArena)
    df["judge_mini"]  = (df["judge_model"] == "GPT-4o-mini").astype(int)
    df["judge_gpt4o"] = (df["judge_model"] == "GPT-4o").astype(int)
    df["judge_llama"] = (df["judge_model"] == "Llama-3.3-70B").astype(int)
    df["judge_qwen"]  = (df["judge_model"] == "Qwen-2.5-VL").astype(int)
    df["bench_work"]  = (df["benchmark"] == "workarena").astype(int)
    df["bench_vwa"]   = (df["benchmark"] == "visualwebarena").astype(int)

    features = [
        "side_effect_bin", "looping_bin", "long_traj",
        "judge_mini", "judge_gpt4o", "judge_llama", "judge_qwen",
        "bench_work", "bench_vwa",
    ]
    feat_labels = [
        "SE-positive (expert)", "Loop-positive (expert)", "Long trajectory",
        "Judge: GPT-4o-mini", "Judge: GPT-4o",
        "Judge: Llama-3.3-70B", "Judge: Qwen-2.5-VL",
        "Benchmark: WorkArena", "Benchmark: VisualWebArena",
    ]

    for outcome, outcome_col in [
        ("Side-effect error", "error_side_effect"),
        ("Looping error",     "error_looping"),
    ]:
        valid = df.dropna(subset=[outcome_col, "side_effect_bin", "looping_bin"])
        X = valid[features].values
        y = valid[outcome_col].values

        # Point estimate
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        point_ors = np.exp(clf.coef_[0])

        # Clustered bootstrap CIs
        traj_keys = valid["traj_key"].values
        unique_trajs = np.unique(traj_keys)
        boot_coefs = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            sampled = rng.choice(unique_trajs, size=len(unique_trajs), replace=True)
            idx = np.concatenate([np.where(traj_keys == t)[0] for t in sampled])
            X_b = X[idx]; y_b = y[idx]
            try:
                clf_b = LogisticRegression(max_iter=500, random_state=0)
                clf_b.fit(X_b, y_b)
                boot_coefs.append(clf_b.coef_[0])
            except Exception:
                continue
        boot_coefs = np.array(boot_coefs)
        ci_lo = np.exp(np.percentile(boot_coefs, 2.5, axis=0))
        ci_hi = np.exp(np.percentile(boot_coefs, 97.5, axis=0))

        # Significance from bootstrap (p approx: proportion of bootstrap CIs crossing 1)
        p_vals = []
        for i in range(len(features)):
            coef_boots = boot_coefs[:, i]
            p_approx = min(
                2 * min(
                    np.mean(coef_boots > 0),
                    np.mean(coef_boots < 0)
                ),
                1.0
            )
            p_vals.append(p_approx)

        print(f"\n  {outcome} (N={len(valid)} rows, "
              f"{len(unique_trajs)} trajectories):")
        print(f"  {'Predictor':<30} {'OR':>7} {'95% CI':>18} {'p':>8}")
        print("  " + "-" * 67)
        for i, (label, or_val, lo, hi, p) in enumerate(
            zip(feat_labels, point_ors, ci_lo, ci_hi, p_vals)
        ):
            print(f"  {label:<30} {or_val:>7.2f} "
                  f"[{lo:>5.2f}, {hi:>6.2f}]  {sig_stars(p):>8}")


# ── Escalation strategies (Figure 1) ─────────────────────────────────────────

def compute_escalation(merged, agent_name):
    """Compute review burden and SE recall for escalation strategies."""
    wide = merged.pivot_table(
        index="traj_key",
        columns="judge_model",
        values="judge_side_effect",
        aggfunc="first",
    ).reset_index()
    expert = merged.drop_duplicates("traj_key")[["traj_key", "side_effect_bin"]]
    wide = wide.merge(expert, on="traj_key").dropna(subset=["side_effect_bin"])
    N = len(wide)
    N_pos = int(wide["side_effect_bin"].sum())

    strategies = {}
    jcols = [j for j in JUDGES if j in wide.columns]
    wide["vote_sum"] = wide[jcols].sum(axis=1)

    def record(label, flag_mask):
        flagged = wide[flag_mask]
        burden  = len(flagged) / N
        tp      = flagged["side_effect_bin"].sum()
        recall  = tp / N_pos if N_pos > 0 else 0
        strategies[label] = (burden, recall)

    for j in jcols:
        record(j, wide[j] == 1)
    record("Majority vote (≥3/5)", wide["vote_sum"] >= 3)
    if "GPT-4o-mini" in wide.columns and "GPT-4o" in wide.columns:
        record("Mini≠GPT-4o disagree",
               wide["GPT-4o-mini"] != wide["GPT-4o"])
    if "GPT-4o" in wide.columns and "Llama-3.3-70B" in wide.columns:
        record("GPT-4o or Llama flags",
               (wide["GPT-4o"] == 1) | (wide["Llama-3.3-70B"] == 1))
    return strategies, N, N_pos


def figure1(agent_data, output_dir):
    """Figure 1: three-panel recall vs burden scatter."""
    strategy_styles = {
        "GPT-4o-mini":           ("o", "#d62728"),
        "GPT-4o":                ("P", "#1f77b4"),
        "Claude-3.7-Sonnet":     ("s", "#2ca02c"),
        "Llama-3.3-70B":         ("v", "#8c564b"),
        "Qwen-2.5-VL":           ("D", "#9467bd"),
        "Majority vote (≥3/5)":  ("^", "#ff7f0e"),
        "Mini≠GPT-4o disagree":  ("X", "#bcbd22"),
        "GPT-4o or Llama flags": ("*", "#17becf"),
    }

    panel_titles = {
        "GPT-4o":  "GPT-4o agent",
        "Claude":  "Claude agent",
        "Qwen":    "Qwen agent",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    for ax, (agent_name, merged) in zip(axes, agent_data.items()):
        strategies, N, N_pos = compute_escalation(merged, agent_name)
        title = f"{panel_titles[agent_name]} (SE={N_pos})"

        for strat_name, (burden, recall) in strategies.items():
            style = strategy_styles.get(strat_name, ("o", "#aaaaaa"))
            marker, color = style
            ax.scatter(burden, recall, marker=marker, color=color,
                       s=110, zorder=3, edgecolors="white", linewidths=0.5)

        # Annotate key strategies
        for strat_name, (burden, recall) in strategies.items():
            if strat_name == "Claude-3.7-Sonnet":
                ax.annotate("Claude", xy=(burden, recall),
                            xytext=(burden + 0.05, recall - 0.09),
                            fontsize=6.5, color="#2ca02c",
                            arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.7))
            if strat_name == "GPT-4o":
                ax.annotate("GPT-4o", xy=(burden, recall),
                            xytext=(burden - 0.22, recall - 0.08),
                            fontsize=6.5, color="#1f77b4",
                            arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.7))
            if strat_name == "GPT-4o-mini" and agent_name != "Qwen":
                ax.annotate("Mini", xy=(burden, recall),
                            xytext=(burden + 0.03, recall - 0.09),
                            fontsize=6.5, color="#d62728",
                            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7))

        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)
        ax.set_xlabel("Review burden", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Side-effect recall", fontsize=8)

    legend_elements = [
        plt.scatter([], [], marker=s[0], color=s[1], s=60, label=name,
                    edgecolors="white", linewidths=0.5)
        for name, s in strategy_styles.items()
    ]
    axes[2].legend(handles=legend_elements, fontsize=6.2, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", loc="lower right",
                   handletextpad=0.4, borderpad=0.5, labelspacing=0.3)

    fig.suptitle("Side-effect recall vs. review burden across evaluated agents",
                 fontsize=9, fontweight="bold", y=1.01)
    plt.tight_layout(w_pad=1.5)

    out = Path(output_dir) / "figure1_recall_burden.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"\nFigure 1 saved to {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce analysis from 'Selecting LLM Judges for "
                    "Agent Evaluation Pipelines: Accuracy Masking and "
                    "Review-Burden Tradeoffs'"
    )
    parser.add_argument("--gpt4o_zip",   required=True,
                        help="Path to GPT-4o agent judgment zip")
    parser.add_argument("--claude_zip",  required=True,
                        help="Path to Claude agent judgment zip")
    parser.add_argument("--qwen_zip",    required=True,
                        help="Path to Qwen agent judgment zip")
    parser.add_argument("--annotations", required=True,
                        help="Path to annotations.csv")
    parser.add_argument("--output_dir",  default="./output",
                        help="Directory for figures and output files")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("LOADING DATA")
    print("=" * 65)

    # GPT-4o agent: bench/agent/judge/file (bench_idx=0, judge_idx=2)
    print("\nGPT-4o agent:")
    gpt4o_j = load_judgments(args.gpt4o_zip, bench_idx=0, judge_idx=2)
    gpt4o_e = load_expert_labels(args.annotations, AGENT_MODELS["gpt4o"])
    gpt4o   = merge_with_expert(gpt4o_j, gpt4o_e)

    # Claude agent: judgments/bench/agent/judge/file (bench_idx=1, judge_idx=3)
    print("\nClaude agent:")
    claude_j = load_judgments(args.claude_zip, bench_idx=1, judge_idx=3)
    claude_e = load_expert_labels(args.annotations, AGENT_MODELS["claude"])
    claude   = merge_with_expert(claude_j, claude_e)

    # Qwen agent: Qwen/bench/agent/judge/file (bench_idx=1, judge_idx=3)
    print("\nQwen agent:")
    qwen_j = load_judgments(args.qwen_zip, bench_idx=1, judge_idx=3)
    qwen_e = load_expert_labels(args.annotations, AGENT_MODELS["qwen"])
    qwen   = merge_with_expert(qwen_j, qwen_e)

    agent_data = {
        "GPT-4o": gpt4o,
        "Claude":  claude,
        "Qwen":    qwen,
    }

    # Tables
    table_accuracy_ranges(agent_data)
    table_se_profiles(agent_data)
    table_appendix_accuracy(agent_data)
    table_appendix_se(agent_data)
    table_appendix_looping(agent_data)
    run_regression(gpt4o)

    # Figure
    figure1(agent_data, args.output_dir)

    print("\n" + "=" * 65)
    print("DONE — all tables printed above, figure saved to output_dir")
    print("=" * 65)


if __name__ == "__main__":
    main()
