import os
from copy import deepcopy

from pydantic import BaseModel

from src.llm.llm_factory import get_llm_instance
from src.utils.file_io import load_json, save_as_json

langs = [
    "afr",
    "arb",
    "ary",
    "arz",
    "asm",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cmn_Hant",
    "cym",
    "dan",
    "deu",
    "ell",
    "est",
    "fin",
    "fra",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "nld",
    "nno",
    "nob",
    "npi",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "spa",
    "srp",
    "swe",
    "swh",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "vie",
    "yue",
    "zlm",
]


def load_nq_data():
    data_path = "/Users/gldong/Documents/latex/ldfighter/code/ldfighter/dataset/translated_nq.json"
    nq_translated = load_json(data_path)
    return nq_translated


class TranslationRating(BaseModel):
    score: int


score_pmt = """You are a translation expert. Given a language code, the original text, and the translated text, you should score the quality of a translation according to the following criteria:
1. Accuracy: Does the translated text convey the same meaning as the original text without adding or omitting key information?
2. Fluency: Is the translation smooth, natural, and grammatically correct in the target language?
3. Cultural Appropriateness: Is the translation sensitive to cultural nuances and does it avoid awkward or inappropriate phrasing?
4. Lexical Choice: Are the words chosen appropriate in terms of meaning and context?
5. Syntax and Structure: Does the translated text maintain the same structure or make sense in the target language without being overly literal or awkward?

Scoring System
- Excellent (9-10): The translation is accurate, fluent, culturally appropriate, and natural, with no errors.
- Good (7-8): The translation is mostly accurate and fluent, with minor issues that don’t affect comprehension.
- Fair (5-6): The translation conveys the general meaning but has noticeable inaccuracies or awkward phrasing.
- Poor (3-4): The translation has significant errors in accuracy, fluency, or cultural relevance, impacting comprehension.
- Unacceptable (1-2): The translation is mostly incorrect or incomprehensible.

Now, let's begin:
Original language: {ori_lang}
Translated language: {trans_lang}
Original text: {text}
Translated text: {translated_text}"""


# Function to load checkpoint
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        checkpoint_data = load_json(checkpoint_file)
        checkpoint_data["processed_questions"] = set(
            checkpoint_data["processed_questions"]
        )
        return checkpoint_data
    else:
        return {
            "processed_questions": set(),
            "language_scores": {},
            "language_counts": {},
            "total_score": 0,
            "total_count": 0,
        }


# Function to save checkpoint
def save_checkpoint(progress, checkpoint_file):
    progress["processed_questions"] = list(progress["processed_questions"])
    save_as_json(progress, checkpoint_file)


def translation_evaluation_other2en():
    llm = get_llm_instance("gpt4json")
    data_path = "/Users/gldong/Documents/latex/ldfighter/code/ldfighter/expe_results/ori_exp_result_nq_advbench/nq/rq1_nq_chatgpt.json_checkpoint_30.json"
    nq_translated = load_json(data_path)
    # Define the checkpoint file
    checkpoint_file = "checkpoint_other2eng.json"

    # Initialize variables from checkpoint
    progress = load_checkpoint(checkpoint_file)

    # Loop through each question and language
    for i, question_reps in enumerate(nq_translated, start=1):
        # Skip the question if it has already been processed
        if i in progress["processed_questions"]:
            continue
        for lang in question_reps:
            if lang not in langs:
                continue
            translated = question_reps[lang]
            original_text = translated["Rep"]
            try:
                translated_text = translated["Rep_En"]
            except KeyError as e:
                print(f"KeyError:{e}, {i}th query", translated)
            # Format the query for translation rating
            query = score_pmt.format(
                ori_lang=lang,
                trans_lang="eng",
                text=original_text,
                translated_text=translated_text,
            )

            # Query the model and get the result
            rst = llm.simple_query(query, response_format=TranslationRating)

            # Extract the score
            score = rst.score
            print(f"Score for {original_text} in {lang}: {score}")

            # Update total score and count for overall average
            progress["total_score"] += score
            progress["total_count"] += 1

            # Update the language-specific scores and counts
            if lang not in progress["language_scores"]:
                progress["language_scores"][lang] = 0
                progress["language_counts"][lang] = 0

            progress["language_scores"][lang] += score
            progress["language_counts"][lang] += 1

        # Mark the question as processed
        progress["processed_questions"].add(i)

        # Save checkpoint after processing each question
        save_checkpoint(deepcopy(progress), checkpoint_file)
        # Checkpoint message
        print(f"Checkpoint: Processed {i}th question ")

    # Calculate overall average score
    overall_average = (
        progress["total_score"] / progress["total_count"]
        if progress["total_count"] > 0
        else 0
    )
    print(f"Overall Average Score: {overall_average}")

    # Calculate average score for each language
    for lang in progress["language_scores"]:
        language_average = (
            progress["language_scores"][lang] / progress["language_counts"][lang]
            if progress["language_counts"][lang] > 0
            else 0
        )
        print(f"Average Score for {lang}: {language_average}")


def translation_evaluation_en2other():
    llm = get_llm_instance("gpt4json")
    data_path = "/Users/gldong/Documents/latex/ldfighter/code/ldfighter/dataset/translated_nq.json"
    nq_translated = load_json(data_path)
    # Define the checkpoint file
    checkpoint_file = "checkpoint.json"

    # Initialize variables from checkpoint
    progress = load_checkpoint(checkpoint_file)

    # Loop through each question and language
    for question in nq_translated:
        # Skip the question if it has already been processed
        if question in progress["processed_questions"]:
            continue
        for lang in nq_translated[question]:
            translated = nq_translated[question][lang]

            # Format the query for translation rating
            query = score_pmt.format(
                ori_lang="eng",
                trans_lang=lang,
                text=question,
                translated_text=translated,
            )

            # Query the model and get the result
            rst = llm.simple_query(query, response_format=TranslationRating)

            # Extract the score
            score = rst.score
            print(f"Score for {question} in {lang}: {score}")

            # Update total score and count for overall average
            progress["total_score"] += score
            progress["total_count"] += 1

            # Update the language-specific scores and counts
            if lang not in progress["language_scores"]:
                progress["language_scores"][lang] = 0
                progress["language_counts"][lang] = 0

            progress["language_scores"][lang] += score
            progress["language_counts"][lang] += 1

        # Mark the question as processed
        progress["processed_questions"].add(question)

        # Save checkpoint after processing each question
        save_checkpoint(deepcopy(progress), checkpoint_file)
        # Checkpoint message
        print(f"Checkpoint: Processed question '{question}'")

    # Calculate overall average score
    overall_average = (
        progress["total_score"] / progress["total_count"]
        if progress["total_count"] > 0
        else 0
    )
    print(f"Overall Average Score: {overall_average}")

    # Calculate average score for each language
    for lang in progress["language_scores"]:
        language_average = (
            progress["language_scores"][lang] / progress["language_counts"][lang]
            if progress["language_counts"][lang] > 0
            else 0
        )
        print(f"Average Score for {lang}: {language_average}")


if __name__ == "__main__":
    translation_evaluation_en2other()
    translation_evaluation_other2en()
