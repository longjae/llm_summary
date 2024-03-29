import argparse
import json
import logging.config
import os

from metric import BatchEvaluation

with open("../config/logging.json", "r") as f:
    config = json.load(f)
    config["handlers"]["file"]["filename"] = "../logs/generation.log"
    logging.config.dictConfig(config)
logger = logging.getLogger()


def batch_evaluation(cot, start_id, end_id):
    file_path = f"../output/{cot}_output.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["output"]

    eva_ori_std = BatchEvaluation()
    eva_ori_cot = BatchEvaluation()

    for i in range(start_id, end_id + 1):
        label = data[i]["abstract"]
        std_pred = data[i]["std_summary"]
        if cot == "t5":
            cot_pred = data[i]["plm_summary"]
        else:
            cot_pred = data[i]["cot_summary"]

        if label == "" or std_pred == "" or cot_pred == "":
            continue

        eva_ori_std.set_text(label, std_pred)
        eva_ori_std.get_rouge_score()
        eva_ori_std.get_bs_score()
        if not cot == None:
            eva_ori_cot.set_text(label, cot_pred)
            eva_ori_cot.get_rouge_score()
            eva_ori_cot.get_bs_score()

    logger.info(f"LABEL VS. GPT-3 STD. SUMMARY:")
    logger.info(f"BATCH SIZE: {eva_ori_std.call_time_rs}")
    logger.info(f"R1: {eva_ori_std.total_r1 / eva_ori_std.call_time_rs}")
    logger.info(f"R2: {eva_ori_std.total_r2 / eva_ori_std.call_time_rs}")
    logger.info(f"RL: {eva_ori_std.total_rl / eva_ori_std.call_time_rs}")
    logger.info(f"BERT_SCORE: {eva_ori_std.total_bs / eva_ori_std.call_time_bs}")

    if not cot == None:
        if cot == "cot":
            logger.info(f"LABEL VS. GPT-3 CoT. SUMMARY:")
        elif cot == "law":
            logger.info(f"LABEL VS. GPT-3 LawCoT. SUMMARY:")
        elif cot == "casebrief":
            logger.info(f"LABEL VS. GPT-3 CasebriefCoT. SUMMARY:")
        elif cot == "t5":
            logger.info(f"LABEL VS. GPT-3 T5. SUMMARY:")
        logger.info(f"BATCH SIZE: {eva_ori_cot.call_time_rs}")
        logger.info(f"R1: {eva_ori_cot.total_r1 / eva_ori_cot.call_time_rs}")
        logger.info(f"R2: {eva_ori_cot.total_r2 / eva_ori_cot.call_time_rs}")
        logger.info(f"RL: {eva_ori_cot.total_rl / eva_ori_cot.call_time_rs}")
        logger.info(f"BERT_SCORE: {eva_ori_cot.total_bs / eva_ori_cot.call_time_bs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--cot", default=None, choices=[None, "cot", "law", "t5"])
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default=999)
    args = parser.parse_args()
    batch_evaluation(cot=args.cot, start_id=args.start_id, end_id=args.end_id)
