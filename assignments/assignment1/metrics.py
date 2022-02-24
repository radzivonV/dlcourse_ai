def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    tp = ((ground_truth == 1) & (prediction == 1)).sum()
    precision = tp / prediction.sum()
    recall = tp / ground_truth.sum()
    accuracy = (ground_truth == prediction).sum()/ground_truth.shape[0]
    f1 = (2*precision*recall)/(precision+recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = (prediction == ground_truth).sum() / ground_truth.shape[0]
    return accuracy
