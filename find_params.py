import os

from dotenv import load_dotenv

from project import find_ab_params

load_dotenv()


def main():
    spread = float(os.environ["SPREAD"])
    min_dist = float(os.environ["MIN_DIST"])
    print(f"spread: {spread}, min_dist: {min_dist}")

    a, b = find_ab_params(spread, min_dist)
    print(f"got a, b params: a={a}; b={b}")
    print(f"const a: comptime_float = {a};")
    print(f"const b: comptime_float = {b};")


if __name__ == "__main__":
    main()
