import glob
import sys

sys.path.append("/root/autodl-tmp/project/ldfighter/src")
from src.utils.help_func import load_json
from src.constant import LLMType, llms
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from src.rqs.myplot import plot_bar, plot_heatmap, add_pd_rows


OUTPUT_PATH = "./"

data_dir = "../expe_results/advbench_manually_labeld"

gloabl_params = {
    "font.weight": "bold",  # the only prop to set bold for the x and y tick label
    "legend.fontsize": 35,
    "axes.labelsize": 35,
    "axes.titlesize": 45,
    "axes.labelweight": "bold",
    "axes.labelcolor": "black",
    "xtick.labelsize": 35,
    "xtick.labelcolor": "black",
    "ytick.labelsize": 35,
    "ytick.labelcolor": "black",
}

legend_param = {
    "ncol": 3,
    "title": None,
    "bbox_to_anchor": (0.55, -0.09),
    "handletextpad": 0,
    "columnspacing": 0,
    "markerscale": 2.0,
}

sns.set(style="white", font="arial", rc=gloabl_params)


def extract_init_rst(data_dir):
    """_summary_

    Args:
        data_dir (_type_): _description_

    Returns:
        rst: rst[question_id][lang] = label
    """
    rst = {}
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for question_file in json_files:
        multilingual_result = load_json(question_file)
        question_id = os.path.splitext(os.path.basename(question_file))[0]
        rst[question_id] = {}
        for ele in multilingual_result:
            lang = ele["lang"]
            if len(ele["label"]) == 0:
                print(question_id + "\n", ele)
            else:
                label = ele["label"][0]
                rst[question_id][lang] = label
                if label != "safe" and ele["text"].startswith(
                    "PROMPT_TRIGERING_CONTENT_MANAGEMENT_POLICY"
                ):
                    print(ele)
                if label != "safe" and "Finish_reason: SAFETY" in ele["text"]:
                    print(ele)
                if label == "safe" and "Finish_reason: OTHER" in ele["text"]:
                    rst[question_id][lang] = "unrelated"
    return rst


def get_each_qst_mljr(rst):
    # statistics
    qst_id_list, jr_lst = [], []
    for qid in rst:
        unsafe_count = 0
        for lang in rst[qid]:
            if rst[qid][lang] == "unsafe":
                unsafe_count += 1
        qst_id_list.append(qid)
        jr_lst.append(unsafe_count / len(rst[qid]))
    return qst_id_list, jr_lst


def mjr():
    pd_col_llm = []
    pd_col_qst = []
    pd_col_jr = []
    for llm_type in [
        LLMType.ChatGPT,
        LLMType.Gemini_pro,
        LLMType.Gemma_7B,
        LLMType.Llama2_13B,
    ]:
        rst = extract_init_rst(data_dir + llm_type)
        qst_id_list, jr_lst = get_each_qst_mljr(rst)
        pd_col_llm.extend([llm_type for i in range(len(qst_id_list))])
        pd_col_qst.extend(qst_id_list)
        pd_col_jr.extend(jr_lst)
        print(f"{llm_type}:{np.mean(jr_lst)}")
    mljr_dicr = {"LLM": pd_col_llm, "Question": pd_col_qst, "MLJR": pd_col_jr}
    pd_data = pd.DataFrame(mljr_dicr)
    f, ax = plt.subplots(figsize=(16, 10))
    sns.boxplot(pd_data, x="LLM", y="MLJR")
    sns.stripplot(pd_data, x="LLM", y="MLJR", s=10)
    ax.set_ylabel("Avg.MJR")
    # ax.xaxis.grid(True)
    # plt.yticks(np.arange(0, 0.6, 0.1))
    # chatgpt:0.12297297297297298
    # gemini-pro:0.006306306306306307
    # gemma-7b:0.26607642124883507
    # llama2-13b:0.03356582388840453
    plt.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_PATH + "mljr.pdf", dpi=600, bbox_inches="tight")


def extract_model_lang_dict_ljr(avg=False, var=False):
    """The resulted data format
             LLM      Lang        JR
        0    chatgpt  eng  0.000000
        1    chatgpt  afr  0.066667
        2    chatgpt  arb  0.033333
        3    chatgpt  ary  0.000000
        4    chatgpt  arz  0.033333
    _type_: _description_
    """
    pd_rows = []
    for llm_type in llms:
        rst = extract_init_rst(os.path.join(data_dir, llm_type))
        lang_jr_dict = defaultdict(list)
        for qustion_id in rst.keys():
            for lang in rst[qustion_id]:
                ljr = 1 if rst[qustion_id][lang] == "unsafe" else 0
                lang_jr_dict[lang].append(ljr)
        for lang in lang_jr_dict:
            ljr = np.mean(lang_jr_dict[lang])
            if len(lang_jr_dict[lang]) != 30:
                print(f"{llm_type}, Number of qst:{len(lang_jr_dict[lang])}")
            pd_row = {"LLM": llm_type}
            pd_row["Lang"] = lang
            pd_row["LJR"] = ljr
            pd_rows.append(pd_row)

    df = pd.DataFrame(columns=["LLM", "Lang", "LJR"])
    add_pd_rows(pd_rows, df)

    avg_df = df.groupby(["Lang"]).agg({"LJR": "mean"}).reset_index()
    var_df = df.groupby(["Lang"]).agg({"LJR": "var"}).reset_index()
    avg_df["LLM"] = "Avg"
    var_df["LLM"] = "Var"
    df_list = [df]
    if avg:
        df_list.append(avg_df)
    if var:
        df_list.append(var_df)
    if avg or var:
        return pd.concat(df_list, ignore_index=True)
    return df


def heatmap_variance_bar():
    """
    1. show the heatmap of JR of each LLM across 74 languages
    2. show the barplot of variance
    """
    df = extract_model_lang_dict_ljr()
    df = df.round(4)
    plot_heatmap(df, fig_name="jr_heatmap", annot=True)

    # top-4 languages
    new_df = df[df["LLM"] == "Avg"]
    sorted_df = new_df.sort_values(by="LJR")
    top4 = sorted_df[:4]
    print(top4)
    print(np.mean(top4["LJR"].values.tolist()))

    # bottom-4 languages
    sorted_df = new_df.sort_values(by="LJR", ascending=False)
    top4 = sorted_df[:4]
    print(top4)
    print(np.mean(top4["LJR"].values.tolist()))
    # calcuate varaince
    variance_df = pd.DataFrame(columns=["LLM", "variance of LJR"])
    for llm in llms + ["Avg"]:
        var = np.var(df[df["LLM"] == llm]["LJR"].values.tolist())
        variance_df.loc[len(variance_df)] = {"LLM": llm, "variance of LJR": var}
    plot_bar(variance_df, "var_ljr")


if __name__ == "__main__":
    # mljr()
    # mmjr()
    heatmap_variance_bar()
    # mjr()
    # high_reource = ["eng", "deu", "fra", "swe", "cmn", "spa", "rus", "nld", "ita", "jpn", "pol", "por", "vie",  "ukr", "kor", "cat", "srp", "ind", "ces","fin", "hun", "nob","ron", "bul", "dan","slv", "hrv"]
    # print(len(high_reource))
