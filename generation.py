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

    for i in range(args.start_id, args.end_id + 1):
        logger.info(f"IDX #: {i}")
        src = data["data"][i]["text"]
        x = (
            f"Article: {src} \n"
            + "Summarize the above article in a sentence in Korean:"
        )
        logger.info(f"INPUT: {x}")
        pred_std = decoder.decode(input=x).content
        logger.info(f"OUTPUT: {pred_std} \n")
        # === 키워드 추출 프롬프트 및 키워드 기반 요약 프롬프트 추가 예정
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
