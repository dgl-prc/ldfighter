import json
import re
import textdistance
import tensorflow.compat.v1 as tf

# from nltk.corpus import stopwords
stop_words = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
    ]
)


def remove_stop_words(strs):
    filtered_words = [w for w in strs if w not in stop_words]
    return set(filtered_words)


def calculate_f1_score(fp, tp, fn):
    precision = 0
    recall = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)

    # Avoid division by zero if precision or recall is 0
    if precision == 0 or recall == 0:
        return (precision, recall, 0)

    f1_score = 2 * (precision * recall) / (precision + recall)
    # print("tp: " + str(tp) + ", fp: " + str(fp) + ", fn: " + str(fn) + ", precision: "  + str(precision) + ", recall: " + str(recall) + ",    f1: " + str(f1_score))
    return (precision, recall, f1_score)


def fuzzy(str1, str2):
    return textdistance.levenshtein.distance(str1, str2) <= 1


def identical(str1, str2):
    return str1 == str2


def match(tokens_g, tokens_p, is_match):
    tps = set()
    fps = set()
    tns = set()
    for token_p in tokens_p:
        matched = False
        for token_g in tokens_g:
            if is_match(token_g, token_p):
                tps.add(token_g)
                matched = True
        if not matched:
            fps.add(token_p)

    tns = tokens_g - tps

    # print(tps)
    # print(fps)
    # print(tns)
    # print()

    return tps, fps, tns


def f1_match_score(prediction, ground_truth):
    ground_truth = re.sub(r"[^\w\s]", "", ground_truth.lower())
    prediction = re.sub(r"[^\w\s]", "", prediction.lower())
    tokens_p = set(prediction.split(" "))
    tokens_g = set(ground_truth.split(" "))
    tokens_p = remove_stop_words(tokens_p)
    tokens_g = remove_stop_words(tokens_g)
    # print(tokens_g)
    # print(tokens_p)

    # tp: number of tokens* that are shared between the correct answer and the prediction.
    # fp: number of tokens that are in the prediction but not in the correct answer.
    # fn: number of tokens that are in the correct answer but not in the prediction.
    tp_set, fp_set, fn_set = match(tokens_g, tokens_p, fuzzy)
    return calculate_f1_score(len(fp_set), len(tp_set), len(fn_set))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    max_i = -1
    max_f1 = -1
    for i in range(len(scores_for_ground_truths)):
        score = scores_for_ground_truths[i]
        if score[-1] > max_f1:
            max_f1 = score[-1]
            max_i = i

    return scores_for_ground_truths[max_i]


def evaluate_predictions(references_path, predictions_path, answer_field="answer"):
    """Calculates and returns metrics."""
    references = {}
    with tf.io.gfile.GFile(references_path) as f:
        for line in f:
            example = json.loads(line)
            references[example["question"]] = example[answer_field]
    #   print("Found {} references in {}".format(len(references), references_path))

    # print("model, lang, question, ground_truth_answer, prediction, precision, recall, f1")
    # print("model, lang, avg_precision, avg_recall, avg_f1")
    #   print("model, avg_precision, avg_recall, avg_f1")
    #   print("lang, avg_precision, avg_recall, avg_f1")
    predictions = {}
    qst_set = set()
    pred_set = set()
    questions = set()
    lang_set = set()
    with tf.io.gfile.GFile(predictions_path) as f:
        avg_dict = {}
        cnt = 0
        for line in f:
            example = json.loads(line)
            qst = example["question"]
            prediction = example["prediction"]
            prediction = prediction.replace("\n", "")
            lang_set.add(example["lang"])
            pred_set.add(prediction + "+" + example["lang"])

            questions.add(qst)
            answers = references[qst]
            metric = metric_max_over_ground_truths(
                metric_fn=f1_match_score, prediction=prediction, ground_truths=answers
            )

            cnt = cnt + 1
            key1 = example["model"]
            if not key1 in avg_dict:
                avg_dict[key1] = {}

            key2 = example["lang"]
            model_dict = avg_dict[key1]
            if not key2 in model_dict:
                model_dict[key2] = [0, 0, 0]

            avg_dict[key1][key2][0] = avg_dict[key1][key2][0] + metric[0]
            avg_dict[key1][key2][1] = avg_dict[key1][key2][1] + metric[1]
            avg_dict[key1][key2][2] = avg_dict[key1][key2][2] + metric[2]

            example["precision"] = metric[0]
            example["recall"] = metric[1]
            example["f1_score"] = metric[2]
            example["prediction"] = ""
            prediction = prediction.replace(",", "")
            print(json.dumps(example))

            print(
                example["model"]
                + ","
                + example["lang"]
                + ","
                + qst
                + ","
                + str(answers).replace(",", "/")[1:-1]
                + ","
                + prediction
                + ","
                + str(metric[0])
                + ","
                + str(metric[1])
                + ","
                + str(metric[2])
            )
        print(len(questions))
        print("," + str(model_dict.keys())[11:-2])
        for model in avg_dict:
            model_dict = avg_dict[model]
            line = model
            for lang in model_dict:
                line = line + ", " + str(model_dict[lang][2] / len(questions))

            print(line)

    # import random

    # pred_set = random.sample(pred_set, 500)
    # for pred in pred_set:
    #     strs = pred.split("+")
    #     obj = {"pred": strs[0], "from": strs[1], "to": random.choice(list(lang_set))}
    #     print(json.dumps(obj))


#   return evaluate_predictions_impl(references, predictions)

evaluate_predictions("expe_results/ground_truth.json", "expe_results/nq.json")
