import sys

sys.path.append("/root/autodl-tmp/ldfighter/src")
from src.rqs.rq1.rq1 import extract_model_lang_dict_ljr
from src.rqs.rq2.rq2 import prepare_f1_df
import pandas as pd
from src.constant import llms, LLMType
import numpy as np
from myplot import (
    plot_bar,
    plot_heatmap,
    add_pd_rows,
    plot_heatmap_correlation,
    plot_heatmap_jumbo,
)

high_reource = [
    "eng",
    "deu",
    "fra",
    "swe",
    "cmn",
    "spa",
    "rus",
    "nld",
    "ita",
    "jpn",
    "pol",
    "por",
    "vie",
    "ukr",
    "kor",
    "cat",
    "srp",
    "ind",
    "ces",
    "fin",
    "hun",
    "nob",
    "ron",
    "bul",
    "dan",
    "slv",
    "hrv",
]


# CI-score
def prepare_comprehensive_index(alph, beta, return_all=False, avg=False, var=False):
    ljr_df = extract_model_lang_dict_ljr()
    f1_df = prepare_f1_df()
    normalized_df = pd.merge(ljr_df, f1_df)
    # ljr_df["F1"] = f1_df["f1_score"]
    columns_to_normalize = ["LJR", "f1_score"]
    for column in columns_to_normalize:
        normalized_df[column] = (
            normalized_df[column] - normalized_df[column].min()
        ) / (normalized_df[column].max() - normalized_df[column].min())
    normalized_df["CI"] = alph * normalized_df["f1_score"] - beta * normalized_df["LJR"]
    ljr_f1_ci_df = normalized_df[normalized_df["LLM"] != "Avg"]
    ci_df = ljr_f1_ci_df.drop(["LJR", "f1_score"], axis=1)
    avg_df = ci_df.groupby(["Lang"]).agg({"CI": "mean"}).reset_index()
    var_df = ci_df.groupby(["Lang"]).agg({"CI": "var"}).reset_index()
    avg_df["LLM"] = "Avg"
    var_df["LLM"] = "var"
    # ci_df = pd.concat([ci_df, avg_df,var_df], ignore_index=True)
    df_list = [ci_df]
    if avg:
        df_list.append(avg_df)
    if var:
        df_list.append(var_df)
    if avg or var:
        ci_df = pd.concat(df_list, ignore_index=True)
    if return_all:
        return ljr_f1_ci_df
    return ci_df


def find_extrem(df, column, llm):
    tmp_df = df[df["LLM"] == llm]
    print(tmp_df.loc[tmp_df[column].idxmax()])


def hight_ration(topK, sorted_list):
    high = [ele for ele in sorted_list[:topK] if ele in high_reource]
    return len(high) / topK


def inspect_gemma7b():
    df = pd.read_json("../expe_results/rq2_f1.json")
    df.rename(columns={"model": "LLM"}, inplace=True)
    df.rename(columns={"lang": "Lang"}, inplace=True)
    new_df = df[(df["Lang"].isin(["rus", "eng"])) & (df["LLM"] == "gemma-7b")]
    new_df.to_excel("output.xlsx", index=False, sheet_name="Sheet1")


def headmap():
    # inspect_gemma7b()
    ci_df = prepare_comprehensive_index(0.5, 0.5)
    plot_heatmap(ci_df, "ci_heatmap", index="LLM", columns="Lang", values="CI")
    common_top_k = set()
    # for llm in llms+["Avg"]:
    for llm in ["Avg"]:
        new_df = ci_df[ci_df["LLM"] == llm]
        sorted_df = new_df.sort_values(by="CI")
        sorted_list = sorted_df["Lang"].values.tolist()
        common_top_k = common_top_k.union(set(sorted_list[:5]))
        # print(llm,sorted_list[:5])
        print(sorted_df[:5])
    # print("=====================")
    # print(common_top_k)
    # print(len(common_top_k))
    # print("common",hight_ration(46,list(common_top_k)))
    #     # print(llm, [:10])
    # print("==========ljr_f1_ci_df=============")
    # print(ljr_f1_ci_df[(ljr_f1_ci_df["LLM"]=="llama2-13b") & (ljr_f1_ci_df["Lang"].isin(["dan","eng"]))])
    # print(ljr_f1_ci_df[(ljr_f1_ci_df["LLM"]=="gemma-7b") & (ljr_f1_ci_df["Lang"].isin(["rus","eng"]))])
    # print("==========ci_df=============")
    # print(ci_df[(ci_df["LLM"]=="llama2-13b") & (ci_df["Lang"].isin(["dan","eng"]))])
    # print(ci_df[(ci_df["LLM"]=="gemma-7b") & (ci_df["Lang"].isin(["rus" ,"eng"]))])


def llm_mean_ci():
    ci_df = prepare_comprehensive_index(0.5, 0.5)
    for llm_type in [
        LLMType.ChatGPT,
        LLMType.Gemini_pro,
        LLMType.Gemma_7B,
        LLMType.Llama2_13B,
    ]:
        ci_list = ci_df[ci_df["LLM"] == llm_type]["CI"].values.tolist()
        mean = np.mean(ci_list)
        print(f"{llm_type}: {mean}")


# calculate the correlations of different languages across different models:
def language_correlatinos(metric, rank=False):
    ljr_f1_ci_df = prepare_comprehensive_index(0.5, 0.5)
    if rank:
        ljr_f1_ci_df["Rank"] = (
            ljr_f1_ci_df.groupby("LLM")[metric]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        result = ljr_f1_ci_df.pivot(index="Lang", columns="LLM", values="Rank")
        method = "spearman"
    else:
        result = ljr_f1_ci_df.pivot(index="Lang", columns="LLM", values=metric)
        method = "pearson"
    result.columns.name = None
    result.index.name = None
    transposed = result.T
    spearman_corr = transposed.corr(method=method)
    # print(spearman_corr[:4])
    plot_heatmap_correlation(spearman_corr, f"ci_cor_heatmap_{metric}_rank_{rank}")
    all_langs = spearman_corr.index.values.tolist()
    high_reource = [
        "eng",
        "deu",
        "fra",
        "swe",
        "cmn",
        "spa",
        "rus",
        "nld",
        "ita",
        "jpn",
        "pol",
        "por",
        "vie",
        "ukr",
        "kor",
        "cat",
        "srp",
        "ind",
        "ces",
        "fin",
        "hun",
        "nob",
        "ron",
        "bul",
        "dan",
        "slv",
        "hrv",
    ]
    low_resource = sorted(list(set(all_langs) - set(high_reource)))
    ## Overall mean
    matrix_values = spearman_corr.to_numpy()
    # 创建掩码，排除对角线
    mask = ~np.eye(matrix_values.shape[0], dtype=bool)
    # 计算对角线以外的值的平均值
    average_correlation = matrix_values[mask].mean()

    ## High resource to High resource
    tmp = []
    for lang1 in high_reource:
        for lang2 in high_reource:
            if lang1 != lang2:
                tmp.append(spearman_corr[lang1][lang2])
                # print((lang1,lang2),spearman_corr[lang1][lang2])
    avg_high_high = np.array(tmp).mean()

    ## low resource to low resource
    tmp = []
    for lang1 in low_resource:
        for lang2 in low_resource:
            if lang1 != lang2:
                tmp.append(spearman_corr[lang1][lang2])
    avg_low2low = np.array(tmp).mean()

    ## low resource to high resource
    tmp = []
    for lang1 in low_resource:
        for lang2 in high_reource:
            if lang1 != lang2:
                tmp.append(spearman_corr[lang1][lang2])
    avg_low2high = np.array(tmp).mean()

    print(
        f"{metric}-Overall average correlation (excluding diagonal): {average_correlation:.4f}"
    )
    print(f"{metric}-average correlation high2high: {avg_high_high:.4f}")
    print(f"{metric}-average correlation low2low: {avg_low2low:.4f}")
    print(f"{metric}-average correlation low2high: {avg_low2high:.4f}")


if __name__ == "__main__":
    llm_mean_ci()
