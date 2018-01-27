import numpy as np, pandas as pd

p_lstm = pd.read_csv("lstm.csv")
conv = pd.read_csv("conv_0100.csv")
nbsvm_71 = pd.read_csv('submission_nb-svm_71.csv')
ngram_75 = pd.read_csv('ngram_075.csv')
submission_bn_fasttext_064 = pd.read_csv('submission_bn_fasttext_064.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()

p_res[label_cols] = 0.40 * p_lstm[label_cols] + 0.13 * conv[label_cols] + 0.14 * nbsvm_71[label_cols] + \
                    0.18 * ngram_75[label_cols] + 0.15 * submission_bn_fasttext_064[label_cols]

#def update1(x):
#        return x

#p_res['toxic'] = p_res['toxic'].apply(lambda x: update1(x))

p_res.to_csv('hybrid_submission.csv', index=False)