import json
import logging.config
import os

from dotenv import load_dotenv

from api_request import Decoder
from arguments import parse_arguments

with open("./config/logging.json", "r") as f:
    config = json.load(f)
    logging.config.dictConfig(config)
logger = logging.getLogger()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_APT_KEY")


def get_llm_summary(args, decoder):
    logger.info("LOAD JSON DATA \n")
    # in_file = os.path.join("./data/train.json")
    in_file = os.path.join("./data/valid.json")
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("./msg/sys.txt", "r") as f:
        sys_txt = f.read()
    with open("./msg/sum.txt", "r") as f:
        sum_txt = f.read()
    if not args.cot == None:
        if args.cot != "t5":
            with open(f"./msg/{args.cot}/kw.txt", "r") as f:
                kw_txt = f.read()
            with open(f"./msg/{args.cot}/cot.txt", "r") as f:
                cot_txt = f.read()
        with open("./output/std_output.json", "r") as f:
            std_output = json.load(f)["output"]

    data_output = {"output": []}
    for i in range(args.start_id, args.end_id + 1):
        logger.info(f"IDX #: {i}")
        src = data["data"][i]["text"].replace("\n", " ")
        # --- std_summary ---
        if args.cot == None:
            x = sys_txt + "\n" + f"Article: {src} \n" + sum_txt
            logger.info(f"INPUT: {x}")
            std_sum = decoder.decode(input=x).content
            logger.info(f"OUTPUT: {std_sum} \n")
        # ---
        if not args.cot == None:
            if args.cot != "t5":
                abstract = data["data"][i]["abstract"]
                std_sum = std_output[i]["std_summary"]
                logger.info(f"ABSTRACT: {abstract}")
                # --- cot_summary
                x = sys_txt + "\n" + f"Article: {src} \n" + kw_txt
                logger.info(f"INPUT: {x}")
                cot_keywords = decoder.decode(input=x).content
                logger.info(f"KEYWORDS: {cot_keywords}")
                cot_input = (
                    f"Document: {src} \n" + f"Information: {cot_keywords}: \n" + cot_txt
                )
                logger.info(f"COT INPUT: {cot_input}")
                pred_cot = decoder.decode(input=cot_input).content
                logger.info(f"COT OUTPUT: {pred_cot} \n")
                # ---
                data_output["output"].append(
                    {
                        "index": i,
                        "text": src,
                        "abstract": abstract,
                        "std_summary": std_sum,
                        "cot_keywords": cot_keywords,
                        "cot_summary": pred_cot,
                    }
                )
            elif args.cot == "t5":
                abstract = data["data"][i]["abstract"]
                std_sum = std_output[i]["std_summary"]
                logger.info(f"ABSTRACT: {abstract}")
                pred_plm = decoder.decoder_for_t5(input=src)
                logger.info(f"{args.cot.upper()}_SUMMARY: {pred_plm}")
                data_output["output"].append(
                    {
                        "index": i,
                        "text": src,
                        "abstract": abstract,
                        "std_summary": std_sum,
                        "plm_summary": pred_plm,
                    }
                )
        else:
            data_output["output"].append(
                {
                    "index": i,
                    "text": src,
                    "abstract": data["data"][i]["abstract"],
                    "std_summary": std_sum,
                }
            )
    if args.cot == None:
        args.cot = "std"
    with open(f"./output/{args.cot}_output.json", "w") as f:
        f.write(json.dumps(data_output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_arguments()
    decoder = Decoder()

    get_llm_summary(args, decoder)
