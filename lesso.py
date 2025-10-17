from math import log

def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
            labels_counts[current_label] += 1
            print("类别统计")


