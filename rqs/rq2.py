import sys

sys.path.append("/root/autodl-tmp/ldfighter/src")
import pandas as pd
from src.rqs.myplot import plot_bar, plot_heatmap
import numpy as np
from src.constant import LLMType, llms
import os


data_path_rq_f1 = "../expe_results/rq2_f1.json"


def prepare_f1_df(avg=False, var=False):
    df = pd.read_json(data_path_rq_f1)
    df.rename(columns={"model": "LLM"}, inplace=True)
    df.rename(columns={"lang": "Lang"}, inplace=True)
    # calculate the average of f1-score
    llm_lang_f1_df = df.groupby(["LLM", "Lang"]).agg({"f1_score": "mean"}).reset_index()
    avg_df = llm_lang_f1_df.groupby(["Lang"]).agg({"f1_score": "mean"}).reset_index()
    avg_df["LLM"] = "Avg"
    var_df = llm_lang_f1_df.groupby(["Lang"]).agg({"f1_score": "var"}).reset_index()
    var_df["LLM"] = "Var"
    df_list = [llm_lang_f1_df]
    if avg:
        df_list.append(avg_df)
    if var:
        df_list.append(var_df)
    if avg or var:
        return pd.concat(df_list, ignore_index=True)
    return llm_lang_f1_df


def heatmap_variance_bar():
    """
    1. show the heatmap of JR of each LLM across 74 languages
    2. show the barplot of variance
    """
    final_df = prepare_f1_df()

    plot_heatmap(
        final_df, fig_name="f1_heatmap", index="LLM", columns="Lang", values="f1_score"
    )
    # calcuate varaince
    variance_df = pd.DataFrame(columns=["LLM", "variance of f1-score"])
    for llm in llms + ["Avg"]:
        var = np.var(final_df[final_df["LLM"] == llm]["f1_score"].values.tolist())
        mean = np.mean(final_df[final_df["LLM"] == llm]["f1_score"].values.tolist())
        print(f"{llm}: {mean}")
        variance_df.loc[len(variance_df)] = {"LLM": llm, "variance of f1-score": var}
    plot_bar(
        variance_df, "var_f1", x="LLM", y="variance of f1-score", value_offset=0.0001
    )


if __name__ == "__main__":
    heatmap_variance_bar()
