import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./LJSpeech-1.1/metadata.csv", sep="|", header=None)

train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)

# 각각 저장
train_df.to_csv("./LJSpeech-1.1/train.csv", sep="|", index=False, header=False)
test_df.to_csv("./LJSpeech-1.1/test.csv", sep="|", index=False, header=False)