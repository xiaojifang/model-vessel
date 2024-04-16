import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

import numpy as np

import pickle
#
# def double_threshold_iteration(img, h_thresh, l_thresh, save=True):
#     print(img.shape)
#     h, w = img.shape
#     img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)
#     bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)
#     gbin = bin.copy()
#     gbin_pre = gbin-1
#     while(gbin_pre.all() != gbin.all()):
#         gbin_pre = gbin
#         for i in range(h):
#             for j in range(w):
#                 if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
#                     neighbors = [gbin[i - 1][j - 1], gbin[i - 1][j], gbin[i - 1][j + 1], gbin[i][j - 1],gbin[i][j + 1], gbin[i + 1][j - 1], gbin[i + 1][j], gbin[i + 1][j + 1]]
#                     if sum(neighbors)>=255*5:
#                         gbin[i][j] = 255
#
#     # if save:
#     #     cv2.imwrite(f"save_picture/bin{index}.png", bin)
#     #     cv2.imwrite(f"save_picture/gbin{index}.png", gbin)
#     return gbin/255



def evaluate(y_true, y_scores, y_auc, dataset):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_auc = np.array(y_auc)
    

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    #  原始的# AUC_ROC = roc_auc_score(y_true, y_scores)

    AUC_ROC = roc_auc_score(y_true, y_auc)
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    plt.plot(fpr, tpr, '-')
    plt.title('ROC curve', fontsize=14)
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))

    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    print(np.unique(y_true))
    print(np.unique(y_pred))

    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    iou = 0
    if float(confusion[1, 1] + confusion[0, 1] + confusion[1, 0]) != 0:
        iou = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1] + confusion[1, 0])
    print("iou: " + str(iou))
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_score(y_true, y_pred)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))
    # Save the results
    file_perf = open(f'./output/{dataset}/performances.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()

    plt.show()

    with open(f'./output/{dataset}/auc.pickle', 'wb') as handle:
        pickle.dump([fpr, tpr], handle, protocol=pickle.HIGHEST_PROTOCOL)
