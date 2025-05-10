import sys

sys.path.append("/root/autodl-tmp/project/ldfighter/src")
from models.embedding import SentenceEmbedding
import torch.nn.functional as F
from sentence_transformers.util import cos_sim
from utils.help_func import load_json, save_as_json
import numpy as np
import time
import torch
import json
from utils.check_jail_break import (
    check_for_autodan_attack_success,
    check_for_gcg_attack_success,
    check_for_advbench_attack_success,
    check_for_nq_false_jailbroken,
)
from rqs.rq1.rq1 import extract_init_rst


def _load_attack_data(data_path):
    data = load_json(data_path)
    questions = []
    for ele in data:
        lang_list = []
        rep_list = []
        for lang in ele:
            lang_list.append(lang)
            rep_list.append(ele[lang]["Rep"])
        questions.append({"lang_list": lang_list, "rep_list": rep_list})
    return questions


def load_validate_adv_reps(data_path, attack_mod):
    new_case_reps = []
    cases_reps = _load_attack_data(data_path)
    for ele in cases_reps:
        if attack_mod == "gcg":
            jailbroken = check_for_gcg_attack_success(ele["rep_list"][0])
        else:
            jailbroken = check_for_autodan_attack_success(ele["rep_list"][0])
        if jailbroken:
            new_case_reps.append(ele)
    return new_case_reps


def load_advbench_data(data_path):
    """
    For the advbench, we use the manually labeled results, and rep_list is the lables of reponses
    """
    questions = []
    lang_list = [
        "eng",
        "fra",
        "rus",
        "spa",
        "ces",
        "swe",
        "deu",
        "dan",
        "nob",
        "nld",
        "pol",
    ]
    data = extract_init_rst(data_path)
    for qid in data.keys():
        query_reps = data[qid]
        rep_list = []
        for lang in lang_list:
            rep_list.append(query_reps[lang])
        questions.append({"lang_list": lang_list, "rep_list": rep_list})
    return questions


def load_nq_data(data_path):
    """
    For the advbench, we use the manually labeled results, and rep_list is the lables of reponses
    """
    questions = []
    lang_list = [
        "eng",
        "fra",
        "rus",
        "spa",
        "ces",
        "swe",
        "deu",
        "dan",
        "nob",
        "nld",
        "pol",
    ]
    data = load_json(data_path)
    for query_reps in data:
        rep_list = []
        for lang in lang_list:
            if "Rep_En" in query_reps[lang]:
                rep_list.append(query_reps[lang]["Rep_En"])
            else:
                rep_list.append(query_reps[lang]["Rep"])
        questions.append({"lang_list": lang_list, "rep_list": rep_list})
    return questions


def strict_jb_check(rep_list, attack_mode):
    check_rst = []
    for rep in rep_list:
        if attack_mode == "gcg":
            rst = check_for_gcg_attack_success(rep)
        elif attack_mode == "autodan":
            rst = check_for_autodan_attack_success(rep)
        elif attack_mode == "advbench":
            rst = check_for_advbench_attack_success(rep)
        elif attack_mode == "nq":
            rst = check_for_nq_false_jailbroken(rep)
        else:
            raise Exception(f"Unknow data:{attack_mode}")
        check_rst.append(rst)
    return check_rst


def is_unsafe_input(rep_list, attack_mode):
    """Check if there are unsafe inputs.
    The input is unsafe if at least one response that refuse to answer
    Args:
        rep_list (_type_): _description_
        attack_mode (_type_): _description_

    Returns:
        _type_: _description_
    """
    jb_list = strict_jb_check(rep_list, attack_mode)
    if (len(jb_list) - np.sum(jb_list)) > 0:
        jb_false_idx = [i for i, jailbroken in enumerate(jb_list) if not jailbroken]
        return True, jb_false_idx
    # print(rep_list, jb_list,len(jb_list),np.sum(jb_list))
    return False, []


def vote(embeddings, num_langs=-1):
    num_langs = len(embeddings) if num_langs == -1 else num_langs
    scores = cos_sim(embeddings, embeddings) * 100
    tmp = []
    for k in range(num_langs):
        sim_list_vk = scores[k].tolist()
        # remove the self-sim
        del sim_list_vk[k]
        avg_sim = np.mean(sim_list_vk)
        tmp.append(avg_sim)
    return np.argmax(tmp)


def step1_select_answer(
    topk, cases_reps, attack_mode, embeding_path, safe_check_only=False
):
    refuse_cnt = 0
    final_answers = []
    for i in range(len(cases_reps)):
        rep_list = cases_reps[i]["rep_list"][:topk]
        has_refuse_rep, refuse_rep_idx = is_unsafe_input(rep_list, attack_mode)
        # in this mode, we only count if the input is detected as unsafe
        if safe_check_only:
            if has_refuse_rep:
                refuse_cnt += 1
                print("======check it=======")
                print(rep_list[refuse_rep_idx[0]])
            continue
        else:
            embeddings = torch.load(f"{embeding_path}/embeddings_{i}.pth")
            """if exists safe rejction."""
            if has_refuse_rep:
                refuse_cnt += 1
                # select a most representative answer
                jb_false_embeddings = torch.stack(
                    [embeddings[i] for i in refuse_rep_idx]
                )
                tmp_rep_id = vote(jb_false_embeddings, len(jb_false_embeddings))
                rep_id = refuse_rep_idx[tmp_rep_id]
            else:
                rep_id = vote(embeddings, topk)
            final_answers.append(cases_reps[i]["rep_list"][rep_id])
    return refuse_cnt, final_answers


def step0_do_embeddings(cases_reps, save_path):
    model = SentenceEmbedding("cpu")
    start = time.time()
    for i, case in enumerate(cases_reps):
        rep_list = case["rep_list"]
        embeddings = model.encode(rep_list)
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        torch.save(embeddings, f"{save_path}/embeddings_{i}.pth")
        print(f"case {i}")
        # handle out of memory
        embeddings = ""
    total_time = time.time() - start
    print(f"End. Total time:{total_time}, Avg time:{total_time / len(cases_reps)}")


def check_final_answer(final_answers):
    cnt = 0
    for rep in final_answers:
        jalibroken = check_for_autodan_attack_success(rep)
        if jalibroken:
            cnt += 1
    print(cnt / len(final_answers))


if __name__ == "__main__":
    safe_check_only = True
    # attack_mode="autodan"
    # attack_mode="gcg"
    # attack_mode = "advbench"
    attack_mode = "nq"

    """Advbench"""
    # data_path = f"/root/advbench/chatgpt"
    # data_path = f"/root/advbench/llama2-13b"
    # data_path = f"/root/advbench/gemma-7b"
    # data_path = f"/root/advbench/gemini-pro"

    """NQ"""
    # data_path="/root/ori_exp_result_nq_advbench/nq/rq1_nq_chatgpt.json_checkpoint_30.json"
    data_path = (
        "/root/ori_exp_result_nq_advbench/nq/rq1_nq_gemma-7b.json_checkpoint_30.json"
    )
    # data_path="/root/ori_exp_result_nq_advbench/nq/rq1_nq_gemini-pro.json_checkpoint_30.json"
    # data_path="/root/ori_exp_result_nq_advbench/nq/rq1_nq_llama2-13b.json_checkpoint_264.json"

    # data_path = f"/root/autodl-tmp/project/ldfighter/exp_results/defense_adv/{attack_mode}/step2_multilingual_rep.json"
    save_path = (
        f"/root/autodl-tmp/project/ldfighter/exp_results/defense_adv/{attack_mode}"
    )

    if attack_mode == "advbench":
        cases_reps = load_advbench_data(data_path)[:100]
    elif attack_mode == "nq":
        cases_reps = load_nq_data(data_path)[:30]
    else:
        cases_reps = load_validate_adv_reps(data_path, attack_mode)[:100]

    # do_embeddings(cases_reps,save_path)
    # for topk in range(3, 11, 1):
    for topk in [3]:
        safe_reject_cnt, final_answers = step1_select_answer(
            topk,
            cases_reps,
            embeding_path=save_path,
            attack_mode=attack_mode,
            safe_check_only=safe_check_only,
        )
        if attack_mode == "nq":
            print(
                f"TopK: {topk}, False alarm rate:{100 * safe_reject_cnt / len(cases_reps):.2f}%"
            )
        else:
            asr = 100 * (len(cases_reps) - safe_reject_cnt) / len(cases_reps)
            print(f"TopK: {topk}, Attack sucess rate:{asr:.2f}%")
