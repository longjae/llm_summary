import json
import logging.config
import os

import chardet
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
    ### 데이터 로드
    logger.info("LOAD JSON DATA \n")
    # in_file = os.path.join("./data/train.json")
    in_file = os.path.join("./data/valid.json")
    with open(in_file, "rb") as f:
        result = chardet.detect(f.read())
        file_encoding = result["encoding"]
    with open(in_file, "r", encoding=file_encoding) as f:
        data = json.load(f)

    ### 텍스트 로드
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
        with open("./output/std_output.json", "rb") as f:
            result = chardet.detect(f.read())
            file_encoding = result["encoding"]
        with open("./output/std_output.json", "r", encoding=file_encoding) as f:
            std_output = json.load(f)["output"]
    if args.method == "costar":
        if args.cot == "casebrief":
            with open(f"./msg/casebrief/costar.txt", "r") as f:
                costar_txt = f.read()
        else:
            with open(f"./msg/costar.txt", "r") as f:
                costar_txt = f.read()

    ### 프롬프트 작성 및 json 저장
    data_output = {"output": []}
    for i in range(args.start_id, args.end_id + 1):
        # --- std_summary ---
        logger.info(f"IDX #: {i}")
        src = data["data"][i]["text"].replace("\n", " ")
        abstract = data["data"][i]["abstract"]
        if args.cot == None:
            if args.method == "costar":
                x = f"# CONTEXT # \nSummary of a legal document below \nDocument: {src} \n\n{costar_txt}"
                logger.info(f"INPUT: {x}")
                decoded_content = decoder.decode(input=x).content
                logger.info(f"OUTPUT: {decoded_content} \n")
            else:
                x = sys_txt + "\n" + f"Document: {src} \n" + sum_txt
                logger.info(f"INPUT: {x}")
                std_sum = decoder.decode(input=x).content
                logger.info(f"OUTPUT: {std_sum} \n")
        # ---
        if not args.cot == None:
            if args.cot != "t5":
                std_sum = std_output[i]["std_summary"]
                logger.info(f"ABSTRACT: {abstract}")
                if args.method is None:
                    # --- cot_summary
                    x = sys_txt + "\n" + f"Document: {src} \n" + kw_txt
                    logger.info(f"INPUT: {x}")
                    cot_keywords = decoder.decode(input=x).content
                    logger.info(f"KEYWORDS: {cot_keywords}")
                    cot_input = (
                        f"Document: {src} \n"
                        + f"Information: {cot_keywords}: \n"
                        + cot_txt
                    )
                    logger.info(f"COT INPUT: {cot_input}")
                    pred_cot = decoder.decode(input=cot_input).content
                    logger.info(f"COT OUTPUT: {pred_cot} \n")
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
                ### CO-STAR 기법 적용
                elif args.method == "costar":
                    x = f"# CONTEXT # \nSummary of a legal document below \nDocument: {src} \n\n{costar_txt}"
                    logger.info(f"INPUT: {x}")
                    decoded_content = decoder.decode(input=x).content
                    logger.info(f"OUTPUT: {decoded_content} \n")
                    try:
                        costar_sum = json.loads(decoded_content)
                    except json.JSONDecodeError as e:
                        print("JSONDecodeError:", e)
                        # 디코딩 오류가 발생한 문자열과 위치를 출력
                        print(
                            "Error at:", decoded_content[e.pos - 10 : e.pos + 10]
                        )  # 오류 주변 20자를 출력
                        raise

                    data_output["output"].append(
                        {
                            "index": i,
                            "text": src,
                            "abstract": abstract,
                            "std_summary": std_sum,
                            "costar_keywords": costar_sum["extract_content"],
                            "costar_summary": costar_sum["summary"],
                        }
                    )
                # ---
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
            if args.method == "costar":
                data_output["output"].append(
                    {
                        "index": i,
                        "text": src,
                        "abstract": abstract,
                        "std_summary": decoded_content,
                    }
                )
            else:
                data_output["output"].append(
                    {
                        "index": i,
                        "text": src,
                        "abstract": abstract,
                        "std_summary": std_sum,
                    }
                )
    if args.cot == None:
        args.cot = "std"
    if args.method == "costar":
        with open(
            f"./output/{args.cot}_costar_output.json", "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(data_output, indent=4, ensure_ascii=False))
    else:
        with open(f"./output/{args.cot}_output.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data_output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_arguments()
    decoder = Decoder()

    get_llm_summary(args, decoder)
