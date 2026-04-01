import os
import pandas as pd
from embedding.trainer import *
# 读取你的数据
# df = pd.read_csv("sample500.csv")
def train_embedding_main():
    # ----------------------------------------------------------------
    # Configure the paths below before running.
    # ----------------------------------------------------------------
    # ROOT_DIR = "/path/to/pois_pandas"   # directory containing parquet.gz batches
    # DF_NAME  = "batch_0001.parquet.gz"  # which batch to train on
    ROOT_DIR = "/path/to/pois_pandas"  # <-- update this
    DF_NAME  = "batch_0001.parquet.gz" # <-- update this
    df = pd.read_parquet(os.path.join(ROOT_DIR, DF_NAME))

    # 数据预处理
    print(f"数据集大小: {len(df)}")
    print(f"列名: {df.columns.tolist()}")


    # 分割训练/验证集
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"\n训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")

    # 第一次训练时加使用这个配置
    # model, trainer = train_poi_encoder(
    #     train_df=train_df,
    #     val_df=val_df,
    #     image_base_path="/home/jupyter/image_data",  # 本地路径
    #     # image_base_path="gs://your-bucket",  # 或GCS路径
    #     output_dir="/home/jupyter/poi_encoder_output",
    #     num_epochs=3,
    #     batch_size=48,
    #     learning_rate=1e-4,
    #     use_lora=False,
    #     is_gcs=False  # 如果使用GCS，设为True
    # )

    # Incremental training (resume from checkpoint)
    model, trainer = train_poi_encoder(
        train_df=train_df,
        val_df=val_df,
        image_base_path="/path/to/image_data",  # <-- update for your batch
        output_dir="./poi_encoder_output",
        num_epochs=3,
        batch_size=48,
        learning_rate=5e-5,
        use_lora=False,
        is_gcs=False,
        resume_from_checkpoint=None,  # e.g. "./poi_encoder_output/checkpoint-5001"
    )

    print("\n训练完成！")