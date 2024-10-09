import numpy as np
from .param import *

def greedy_decode(out):
    '''
    Post process of LPRNet
    '''
    out = out.cpu().detach().numpy()
    preb_labels = []
    for i in range(out.shape[0]):
        preb = out[i, :, :]
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    result = []
    for i, label in enumerate(preb_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        result.append(lb)
    
    return result

    for i, label in enumerate(preb_labels):
        # show image and its predict label
        if True:
            show(imgs[i], label, targets[i])
        if len(label) != len(targets[i]):
            Tn_1 += 1
            continue
        if (np.asarray(targets[i]) == np.asarray(label)).all():
            Tp += 1
        else:
            Tn_2 += 1
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print(Info(f"Test Accuracy: {Acc} [{Tp}:{Tn_1}:{Tn_2}:{(Tp+Tn_1+Tn_2)}]"))
    t2 = time.time()
    print(Info(f"Test Speed: {(t2 - t1) / len(datasets)}s 1/{len(datasets)}]"))


def nms():
    '''
    Post process of object detection
    '''
    return 'nms'