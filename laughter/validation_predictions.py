"""
Works now for everything, validate using the IoU to find if a prediction is true or not, then calculate the P,R,F1,ACC.
"""
import glob
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("/home/user/standup_comedyclub/src"))
from normalize_name import normalize_filename

def calculate_iou(interval1, interval2):
    """
    Calculate Intersection over Union (IoU) for two intervals.
    Each interval is a tuple (start, end).
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    area1 = end1 - start1
    area2 = end2 - start2
    union = area1 + area2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union

def evaluate_predictions_with_iou(predictions, labels, iou_threshold=0.5):
    """
    Evaluate predictions against labels using IoU.
    Returns counts of True Positives (TP), False Positives (FP), and False Negatives (FN).
    """
    TP = 0
    FP = 0
    FN = 0

    # Track which labels have been matched
    matched_labels = set()
    not_matched_labels = set()

    # Check each prediction
    for pred in predictions:
        best_iou = 0.0
        best_label_idx = -1

        for i, label in enumerate(labels):
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_labels.add(best_label_idx)  # Mark this label as matched
        else:
            FP += 1
            # not_matched_labels.add(label)

    # Check for FN
    for i, label in enumerate(labels):
        best_iou = 0.0
        for pred in predictions:
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou < iou_threshold:
            not_matched_labels.add(label)

    # Count False Negatives (labels not matched by any prediction)
    # FN = len(labels) - len(matched_labels)
    FN = len(not_matched_labels)

    return TP, FP, FN, not_matched_labels

def _test():
    # Example usage
    predictions = [(1, 3), (5, 7), (9, 11)]  # List of prediction intervals
    labels = [(2, 4), (6, 8), (10, 12)]       # List of label intervals
    iou_threshold = 0.5  # Set the IoU threshold

    TP, FP, FN = evaluate_predictions_with_iou(predictions, labels, iou_threshold)
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")

def calculate_metrics(TP, FP, FN):
    """
    Calculate precision, recall, and accuracy given TP, FP, and FN.
    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1 = 2*TP/(2*TP+FP+FN)

    return precision, recall, accuracy, f1

def read_K_annot(tsv_file):

    df = pd.read_csv(tsv_file, sep="\t", header=None)

    # Filter rows where the first column is "laughter"
    laughter_df = df[df[0] == "laughter"]

    # Extract start and end times
    start_times = laughter_df[2].tolist()
    end_times = laughter_df[3].tolist()

    # Create a new DataFrame in the desired format
    return pd.DataFrame([start_times, end_times], index=["start_sec", "end_sec"])

def df2list(df):
    try:
        dump = list(zip(df.T['start_sec'], df.T['end_sec']))
    except: 
        dump = list(zip(df['start_sec'], df['end_sec']))
    return dump

def validation_test_set(iou_threshold=0.3, 
                        path_pred='/path/to/predictions/es/',
                        path_gt='/path/to/ground/truth/es/',
                        type_pred = 'filtered_candidates',
                        verbose=True):
    """
    type_pred in filtered_candidates (for our model), initial_laughters (for Omine et al., 2024)
    """
        
    TP_all = 0
    FP_all = 0
    FN_all = 0

    list_test_set = [k.split('/')[-1].split('.')[0] for k in glob.glob(path_gt + '*.csv')]

    for fn in list_test_set:
        # csv files with columns "start_sec" and "end_sec"
        
        dfpred = pd.read_csv(path_pred + fn + '.csv')
        dfpred.rename(columns ={'t0': 'start_sec', 't1' : 'end_sec'}, inplace=True)
        if type_pred == "filtered_candidates":
            dfpred = dfpred[dfpred.label == 'risa']
        elif type_pred == "initial_laughters":
            dfpred = dfpred[dfpred.source == 'Initial']

        dfgt = pd.read_csv(path_gt + fn + '.csv')
        if len(dfgt.columns) < 2:
            dfgt = pd.read_csv(path_gt + fn + '.csv', sep=';')
        dfgt.rename(columns ={'t0': 'start_sec', 't1' : 'end_sec'}, inplace=True)

        TP, FP, FN, not_matched_labels = evaluate_predictions_with_iou(df2list(dfpred), df2list(dfgt), iou_threshold=iou_threshold)
        TP_all += TP
        FP_all += FP
        FN_all += FN

    precision, recall, accuracy, f1= calculate_metrics(TP_all, FP_all, FN_all)   
    if verbose:
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1: {f1:.2f}")
        print('\n')

    return TP_all, FP_all, FN_all
    
