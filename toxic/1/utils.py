import numpy as np, pandas as pd

csv = pd.read_csv("submission_bn_fasttext_train_064.csv")
csv.drop("comment_text", axis=1, inplace=True)
csv.to_csv('submission_bn_fasttext_train_064.csv', index=False)