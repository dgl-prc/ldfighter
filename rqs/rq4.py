import pandas as pd
from rqs.myplot import plot_line


# def plot_bar(df,fig_name,x="LLM",y="variance of LJR",value_offset=None)
def prepare_df():
    df = pd.read_csv("/root/autodl-tmp/ldfighter/expe_results/rq4_rq5_vote.csv")
    return df


if __name__ == "__main__":
    # df = prepare_df()
    # print(df)
    # xticklabels = [i for i in range(3,31,3)]
    # xticklabels = ["eng"]+xticklabels
    # print(xticklabels)
    # chatgpt:0.12297297297297298
    # plot_line(df,fig_name="f1_vote",x="top_k",y="avg_f1",hue="LLM",xlabel="k")

    asr = [
        9.68,
        4.30,
        4.30,
        4.30,
        3.23,
        3.23,
        3.23,
        3.23,
        31.00,
        29.00,
        22.00,
        20.00,
        17.00,
        15.00,
        14.00,
        10.00,
    ]
    asr = [e / 100 for e in asr]
    topks = [3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 7, 8, 9, 10]
    attack = ["CGC" for i in range(3, 11, 1)] + ["AutoDAN" for i in range(3, 11, 1)]
    df = pd.DataFrame({"top_k": topks, "ASR": asr, "Attack": attack})
    plot_line(df, fig_name="asr_ldf", x="top_k", y="ASR", hue="Attack", xlabel="k")
