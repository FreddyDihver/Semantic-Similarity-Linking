import numpy as np
from sentence_transformers import util
from sklearn.metrics import precision_recall_curve

def precision_recall_curve_custom(labels, similarity, thresholds):
    """
    Compute the precision-recall curve for custom threshold values
    
    :param labels: Ground truth labels
    :param similarity: Similarity matrix
    :param thresholds: List of custom threshold values
    :return: Precision, recall, and threshold arrays
    """
    labels = np.asarray(labels).astype(int)
    similarity = np.asarray(similarity, dtype=float)
    thres_arr = np.asarray(sorted(thresholds), dtype=float)  # ascending
    prec_arr = []
    rec_arr = []

    # loop through each threshold
    for t in thres_arr:
        pred = (similarity >= t).astype(int)

        # compute true positives, false positives, false negatives
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))

        # compute precision and recall
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # append precision, recall, and threshold values
        prec_arr.append(p)
        rec_arr.append(r)

    return np.array(prec_arr), np.array(rec_arr), np.array(thres_arr)


def F1_fn(model, o_sentences, t_sentences, labels, threshold=0.9, maximize_F1=False, maximize_F1_prec=False,
          target_prec=0.95, batch_size=128, custom_prec_on=True):
    """
    Returns F1, precision, recall, diagonal matrix of true labels and similarity matrix

    :param model: The model
    :param o_sentences: Series of original sentences to compare
    :param t_sentences: Series of target sentences to compare
    :param labels: Truth labels, corresponding to sentences
    :param threshold: Threshold for predicted labels
    :return: F1, precision, recall, predictions, final threshold
    """
    # Ground truth
    N = len(labels)
    labels = np.array(labels)

    # Predicted labels
    o_sentences = o_sentences.tolist()
    o_emb = model.encode(o_sentences,
                         batch_size=batch_size,
                         device="cpu",
                         show_progress_bar=True,
                         convert_to_numpy=True)
    print('Sentences from original dataset encoded')
    t_sentences = t_sentences.tolist()
    t_emb = model.encode(t_sentences,
                         batch_size=batch_size,
                         device="cpu",
                         show_progress_bar=True,
                         convert_to_numpy=True)
    print('Sentences from target dataset encoded')

    # Calculate similarity
    similarity = util.pairwise_cos_sim(o_emb, t_emb).cpu().numpy()
    print('Calculated similarity matrix')

    if not maximize_F1 and not maximize_F1_prec:
        # Predictions based on threshold
        pred = (similarity >= threshold).astype(int)

        # Confusion
        TP = np.sum((pred == 1) & (labels == 1))
        FP = np.sum(pred) - TP
        FN = np.sum((pred == 0) & (labels == 1))

        # Calculate and print metrics
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        F1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    elif maximize_F1_prec:
        print('Calculating optimal treshold')
        if custom_prec_on:
            custom_thresholds = np.linspace(0.2, 1, 80)
            p, r, thres_arr = precision_recall_curve_custom(labels, similarity, custom_thresholds)
        else:
            prec_arr, rec_arr, thres_arr = precision_recall_curve(labels, similarity)

            # Align with thresholds
            p = prec_arr[:-1]
            r = rec_arr[:-1]

        F1_list = 2 * (p * r) / (p + r)
        mask = p >= target_prec

        if np.any(mask):
            cand_idx = np.where(mask)[0]
            best_idx = cand_idx[np.argmax(F1_list[cand_idx])]
        else:
            print("Precision of", target_prec, "could not be reached; using max precision point")
            best_idx = np.argmax(p)
        opt_thres = thres_arr[best_idx]
        pred = (similarity >= opt_thres).astype(int)
        prec = p[best_idx]
        rec = r[best_idx]
        F1 = F1_list[best_idx]
        print("\nOptimal threshold: ", opt_thres)
    else:
        print('Calculating optimal treshold')
        if custom_prec_on:
            custom_thresholds = np.linspace(0.2, 1, 80)
            p, r, thres_arr = precision_recall_curve_custom(labels, similarity, custom_thresholds)
        else:
            prec_arr, rec_arr, thres_arr = precision_recall_curve(labels, similarity)

            # Align with thresholds
            p = prec_arr[:-1]
            r = rec_arr[:-1]

        F1_list = 2 * (p * r) / (p + r)

        best_idx = np.argmax(F1_list[~np.isnan(F1_list)])  # align with thresholds length
        opt_thres = thres_arr[~np.isnan(F1_list)][best_idx]
        pred = (similarity >= opt_thres).astype(int)
        prec = p[~np.isnan(F1_list)][best_idx]
        rec = r[~np.isnan(F1_list)][best_idx]
        F1 = F1_list[~np.isnan(F1_list)][best_idx]
        print("\nOptimal threshold: ", opt_thres)

    print("\nRecall: ", rec)
    print("Precision: ", prec)
    print("F1: ", F1)

    # Return metrics and label variables
    if not maximize_F1 and not maximize_F1_prec:
        return F1, prec, rec, pred, threshold
    else:
        return F1, prec, rec, pred, opt_thres