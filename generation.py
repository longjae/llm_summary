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

    data_output = {"output": []}
    sys_txt = "I want you to act as a expert for summarization for Korean language"
    sum_txt = "Summarize the above article in a sentence in Korean."
    kw_txt = """
    What are the important entities in this document?
    What are the important dates in this document?
    What events are happening in this events?
    What is the result of these events?
    Please answer to JSON the above questions in Korean.
    """
    # You must answer like an example below.
    # example:
    # {
    #     'entities': 'Mr. Baker, the car driver, and the motorcyclist who was injured',
    #     'dates': '4 June and the present day',
    #     'events': 'a collision between Mr. Baker’s motorcycle and a car, and the investigation into the collision',
    #     'result': 'Mr. Baker died and the car driver and motorcyclist were injured.'
    # }
    # An example in english but you must answer in Korean.
    cot_txt = """
    Let's integrate the above information and summarize the article in Korean.
    """
    for i in range(args.start_id, args.end_id + 1):
        logger.info(f"IDX #: {i}")
        src = data["data"][i]["text"]
        x = sys_txt + "\n" + f"Article: {src} \n" + sum_txt
        logger.info(f"INPUT: {x}")
        pred_std = decoder.decode(input=x).content
        logger.info(f"OUTPUT: {pred_std} \n")
        # 테스트 필요
        if args.cot_true:
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
        with open("./output/train_output.json", "w") as g:
            g.write(json.dumps(data_output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_arguments()
    decoder = Decoder()

    get_llm_summary(args, decoder)
