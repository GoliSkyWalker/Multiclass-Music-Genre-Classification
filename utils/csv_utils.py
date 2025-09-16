import pandas as pd

def read_aggregated_features(args):
    """
    Reads the CSV file from data/features/aggregated and returns the DataFrame.
    Throws an error if experiment type is 'svm' or 'nn' and features are 'sequential'.
    """
    # Check for invalid combination
    if args.architecture in ['svm', 'nn'] and getattr(args, 'features', None) == 'sequential':
        raise ValueError("'sequential' features are not supported for SVM or NN architectures.")
    csv_path = 'data/features/aggregated/genrefeatures.csv'
    return pd.read_csv(csv_path)
