import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SumCoT for kor text")
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default="0")
    args = parser.parse_args()

    return args
