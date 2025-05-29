import pandas as pd
import argparse

"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas fastparquet
"""
def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('--file_path', type=str, default='episode_000000.parquet',
                        help='The path to the parquet file')
    args = parser.parse_args()

    # 读取parquet文件
    df = pd.read_parquet(args.file_path)

    # 显示数据基本信息
    print("\n数据基本信息:")
    print(df.info())

    print("\n前5行数据:")
    print(df.head())

    print("\n数据列名:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()