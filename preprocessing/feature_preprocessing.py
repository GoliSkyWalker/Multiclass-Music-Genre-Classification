# will load the extrated features for preprocessing, and save them in data/features/aggregated_features.csv 
import numpy as np
import librosa
import os
import pandas as pd

def concat_wavenet_features(fname):
    encoding = np.load(fname)
    encoding = encoding.reshape((-1, 16))
    if encoding.shape[0] != 16 and encoding.shape[1] == 16:
        pass
    elif encoding.shape[0] == 16 and encoding.shape[1] != 16:
        encoding = encoding.T
    elif encoding.shape[0] == 16 and encoding.shape[1] == 16:
        pass
    else:
        print(f"Warning: Unexpected encoding shape {encoding.shape}. Assuming (timesteps, features).")
    num_timesteps = encoding.shape[0]
    num_channels = encoding.shape[1]
    if num_timesteps == 0:
        return np.zeros(num_channels * 3)
    stddev_wavenet = np.std(encoding, axis=0)
    mean_wavenet = np.mean(encoding, axis=0)
    average_difference_wavenet_sum = np.zeros((num_channels,))
    num_pairs_summed = 0
    if num_timesteps >= 2:
        for i in range(0, num_timesteps - 2, 2):
            average_difference_wavenet_sum += encoding[i] - encoding[i+1]
            num_pairs_summed += 1
    average_difference_wavenet = np.zeros((num_channels,))
    if num_pairs_summed > 0:
        average_difference_wavenet = average_difference_wavenet_sum / num_pairs_summed
    concat_features_wavenet = np.hstack((stddev_wavenet, mean_wavenet, average_difference_wavenet))
    return concat_features_wavenet

def concat_mfcc_features(fname, mfcc_size=13):
    mfcc=np.load(fname)
    stddev_mfccs = np.std(mfcc, axis=1)
    mean_mfccs = np.mean(mfcc, axis=1)
    num_mfcc_frames = mfcc.shape[1]
    mfcc_diff_sum = np.zeros((mfcc_size,))
    mfcc_pairs_counted = 0
    if num_mfcc_frames >= 2:
        for i in range(0, num_mfcc_frames - 2, 2):
            mfcc_diff_sum += mfcc[:, i] - mfcc[:, i+1]
            mfcc_pairs_counted += 1
    average_difference = np.zeros((mfcc_size,))
    if mfcc_pairs_counted > 0:
        average_difference = mfcc_diff_sum / mfcc_pairs_counted
    concat_features_mfcc = np.hstack((stddev_mfccs, mean_mfccs, average_difference))
    return concat_features_mfcc

def list_folders(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def list_files(directory):
    return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]


if __name__ == "__main__":
    # when the function is run directly, it will read all the features, preprocess them and save them in a csv file


    # first check if feature directories exist, if not call feature extraction
    if not os.path.exists('../data/features/sequential/wavenet'):
        raise FileNotFoundError("Wavenet feature directory not found. Please run feature extraction first.")
    if not os.path.exists('../data/features/sequential/mfcc'):
        raise FileNotFoundError("MFCC feature directory not found. Please run feature extraction first.")
    genres = list_folders('../data/features/sequential/wavenet')
    mfcc_size = 13
    dataset = []
    y = []

    # load features and preprocess them
    for genreindex, genre in enumerate(genres):
        wavenet_files = list_files(f'../data/features/sequential/wavenet/{genre}')
        mfcc_files = list_files(f'../data/features/sequential/mfcc/{genre}')
        wavenet_files = [s for s in wavenet_files if '_' not in s]
        mfcc_files = [s for s in mfcc_files if '_' not in s]
        common_files = set(wavenet_files).intersection(set(mfcc_files))

        #if no common files, feature extraction was not run properly
        if len(common_files) == 0:
            raise ValueError(f"No common files found for genre {genre}. Please ensure feature extraction was run correctly.")      
        
        for file in common_files:
            wavenet_path = f'../data/features/sequential/wavenet/{genre}/{file}'
            mfcc_path = f'../data/features/sequential/mfcc/{genre}/{file}'
            wavnet_feat = concat_wavenet_features(wavenet_path)
            mfcc_feat = concat_mfcc_features(mfcc_path, mfcc_size=mfcc_size)
            # Concatenate all features for plotting

            dataset += [(file, wavnet_feat, mfcc_feat )]
            y.append(genreindex)


    labels =[]
    for index in y:
        labels.append(genres[index])
    
    assert len(dataset) == len(labels)
    modified_list = [t + (new_value,) for t, new_value in zip(dataset, labels)]
    df = pd.DataFrame(modified_list, columns=['file', 'wavnet', 'mfcc', 'class'])


    df['wavnet'] = df['wavnet'].apply(np.array2string)
    df['mfcc'] = df['mfcc'].apply(np.array2string)


    df.to_csv('../data/features/aggregated/genrefeatures.csv', index=False)
    print("Feature preprocessing complete. Features saved to genrefeatures.csv")
