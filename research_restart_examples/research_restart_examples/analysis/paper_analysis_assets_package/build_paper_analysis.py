from pathlib import Path
import os, glob, json, textwrap
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Edit these paths if needed
BASE_ANALYSIS = Path("analysis")
LING_PATH = BASE_ANALYSIS / "linguistics" / "suite_20260404T090445Z_factorial" / "linguistics_final_runs.csv"
TOK_LONG_PATH = BASE_ANALYSIS / "tokenizers" / "suite_20260404T090445Z_factorial" / "tokenizer_counts_long.csv"
OUTPUTS_BASE = Path("outputs") / "suite_20260404T090445Z_factorial"
OUT_DIR = Path("paper_analysis_assets")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LING_PATH)
tok = pd.read_csv(TOK_LONG_PATH)
ref = tok[tok["tokenizer_name"]=="reference::openai-community/gpt2"][["run_dir","token_count"]].rename(columns={"token_count":"reference_gpt2_tokens"})
df = df.merge(ref, on="run_dir", how="left")

engine_order=["rtsm","srpf","lvtc","agc"]
model_order=["distilgpt2","gpt2","qwen2.5-0.5b-instruct"]

engine_summary = df.groupby("engine").agg(
    runs=("engine","size"),
    mean_words=("word_count","mean"),
    sd_words=("word_count","std"),
    mean_mattr_50=("mattr_50","mean"),
    mean_mtld=("mtld","mean"),
    mean_distinct_2=("distinct_2","mean"),
    mean_repeated_trigram_fraction=("repeated_trigram_fraction","mean"),
    mean_prompt_tfidf_cosine=("prompt_tfidf_cosine","mean"),
    mean_previous_step_tfidf_cosine=("previous_step_tfidf_cosine","mean"),
    mean_flesch_reading_ease=("flesch_reading_ease","mean"),
    mean_reference_gpt2_tokens=("reference_gpt2_tokens","mean"),
    truncation_rate=("native_prompt_truncated","mean"),
).reindex(engine_order).round(4)
engine_summary.to_csv(OUT_DIR / "engine_summary.csv")

metrics = ["word_count","mattr_50","mtld","distinct_2","repeated_trigram_fraction","prompt_tfidf_cosine","previous_step_tfidf_cosine","flesch_reading_ease"]
anova_rows=[]
for metric in metrics:
    model = ols(f"{metric} ~ C(engine) * C(model_key) + C(prompt_index)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    ss_total = anova["sum_sq"].sum()
    for term,row in anova.iterrows():
        if term=="Residual":
            continue
        anova_rows.append({"metric":metric,"term":term,"df":row["df"],"F":row["F"],"p_value":row["PR(>F)"],"eta_sq":row["sum_sq"]/ss_total})
pd.DataFrame(anova_rows).to_csv(OUT_DIR / "anova_results.csv", index=False)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8.5,5.2))
sns.boxplot(data=df, x="engine", y="word_count", order=engine_order)
plt.xlabel("Engine"); plt.ylabel("Final output length (words)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_engine_output_length_boxplot.png", dpi=220); plt.close()

plt.figure(figsize=(8.5,5.2))
sns.boxplot(data=df, x="engine", y="prompt_tfidf_cosine", order=engine_order)
plt.xlabel("Engine"); plt.ylabel("TF-IDF cosine to seed prompt")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_engine_prompt_similarity_boxplot.png", dpi=220); plt.close()
