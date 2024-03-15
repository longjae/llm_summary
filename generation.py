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
    in_file = os.path.join("./data/train.json")
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("./msg/sys.txt", "r") as f:
        sys_txt = f.read()
    with open("./msg/sum.txt", "r") as f:
        sum_txt = f.read()
    if args.cot == "cot":
        with open("./msg/cot/kw.txt", "r") as f:
            kw_txt = f.read()
        with open("./msg/cot/cot.txt", "r") as f:
            cot_txt = f.read()
    elif args.cot == "law":
        with open("./msg/law/kw.txt", "r") as f:
            kw_txt = f.read()
        with open("./msg/law/cot.txt", "r") as f:
            cot_txt = f.read()

    data_output = {"output": []}
    for i in range(args.start_id, args.end_id + 1):
        logger.info(f"IDX #: {i}")
        src = data["data"][i]["text"]
        # --- std_summary ---
        x = sys_txt + "\n" + f"Article: {src} \n" + sum_txt
        logger.info(f"INPUT: {x}")
        pred_std = decoder.decode(input=x).content
        logger.info(f"OUTPUT: {pred_std} \n")
        # ---
        if not args.cot == None:
            # --- cot_summary
            x = sys_txt + "\n" + f"Article: {src} \n" + kw_txt
            logger.info(f"INPUT: {x}")
            cot_keywords = decoder.decode(input=x).content
            logger.info(f"KEYWORDS: {cot_keywords}")
            cot_input = (
                f"Article: {src} \n" + f"Information: {cot_keywords}: \n" + cot_txt
            )
            logger.info(f"COT INPUT: {cot_input}")
            pred_cot = decoder.decode(input=cot_input).content
            logger.info(f"COT OUTPUT: {pred_cot} \n")
            # ---
            data_output["output"].append(
                {
                    "index": i,
                    "text": src,
                    "abstract": data["data"][i]["abstract"],
                    "std_summary": pred_std,
                    "cot_keywords": cot_keywords,
                    "cot_summary": pred_cot,
                }
            )
        else:
            data_output["output"].append(
                {
                    "index": i,
                    "text": src,
                    "abstract": data["data"][i]["abstract"],
                    "std_summary": pred_std,
                }
            )
        with open(f"./output/{args.cot}_output.json", "w") as f:
            f.write(json.dumps(data_output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_arguments()
    decoder = Decoder()

    get_llm_summary(args, decoder)
