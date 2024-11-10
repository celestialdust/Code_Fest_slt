from signjoey.metrics import bleu
import pandas as pd
import numpy as np
import sys

GT = pd.read_csv("/media/xins/xinS1/Coding_Fest/GT_test_ref.csv")
hyp = pd.read_csv("/media/xins/xinS1/Coding_Fest/Coding_Fest_slt/sign_SP_H_B_RNN_L3/test_hyp.csv")
hyp.index = np.arange(len(hyp))
print(GT.shape, hyp.shape)
video_sorted_name = GT["Video_Clip_Name"]
gt_subtitle = GT["Inference"]
matching_inference = []
for vn in video_sorted_name:
    flag = False
    for idx in range(len(hyp)):
        cvn = hyp.loc[idx, "Video_Clip_Name"]
        if cvn == vn:
            matching_inference.append(hyp.loc[idx, "Inference"])
            flag = True
    if flag == False:
        print("Error: Current not in reference!")
        print(vn)
        # sys.exit()

# assert(len(matching_inference) == len(gt_subtitle))

bleu_score = bleu(references=gt_subtitle, hypotheses=matching_inference)
print("Result:")
print(bleu_score)