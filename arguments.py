import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SumCoT for kor text")
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default="0")
    parser.add_argument("--cot_true", type=bool, default=False)
    args = parser.parse_args()

    return args
