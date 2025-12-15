import pandas as pd
import numpy as np
from decimal import Decimal
import re

def preprocess_poi_data(df):

    # A. Coordinate Conversion
    df['latitude'] = df['latitude'].apply(lambda x: float(x) if x is not None else 0.0)
    df['longitude'] = df['longitude'].apply(lambda x: float(x) if x is not None else 0.0)
    
    # B. Null Imputation
    df['type_code'] = df['type_code'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else (x if isinstance(x, list) else []))

    def fill_description(row):
        if pd.isna(row['description']) or row['description'] == '':
            types = ", ".join([t.replace('_', ' ').lower() for t in row['type_code']])
            return f"{row['name']} is a {types}."
        return row['description']

    df['description_filled'] = df.apply(fill_description, axis=1)    
    df['brief_description'] = df['brief_description'].fillna(df['description_filled'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x))
    df['media_url_list'] = df['media_url_list'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else (x if isinstance(x, list) else []))
    df['media_url_list'] = df['media_url_list'].apply(lambda x: x if x is not None else [])

    # C. Feature Engineering
    df['reviews_list'] = df['review_entities'].apply(
        lambda x: [r.strip() for r in x.split('||') if r.strip()] if pd.notna(x) else []
    )
    # Parse administrative_code into hierarchical levels
    def parse_admin_code(code):
        if pd.isna(code):
            return pd.Series([None] * 5)
        parts = code.replace('IDN.', '').split('.')
        parts = parts + [None] * (5 - len(parts)) 
        return pd.Series(parts[:5])

    admin_cols = ['admin_L1', 'admin_L2', 'admin_L3', 'admin_L4', 'admin_L5']
    df[admin_cols] = df['administrative_code'].apply(parse_admin_code)

    # D. Final Data Preparation
    # Construct a "Full Text" field for the Embedding model
    # Concatenate name, type, and description as the complete input for the Text Encoder
    df['text_for_embedding'] = df.apply(
        lambda row: f"[NAME] {row['name']} [TYPE] {' '.join(row['type_code'])} [DESC] {row['description_filled']}", 
        axis=1
    )

    print(f"数据预处理完成。原始条目: {len(df)}, 处理后条目: {len(df)}")
    print(f"填补了 {df['description'].isna().sum()} 个缺失的 description。")
    
    return df

if __name__ == "__main__":
    import numpy as np
    df = pd.read_csv('../dataset/id_pois/sample500_raw.csv')
    # data preprocessing
    clean_df = preprocess_poi_data(df)