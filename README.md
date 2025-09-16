

# Shallow aggregation and sequential classification of music genres
**steps to run this project**

> **Hint:** For best reproducibility, use the provided `environment.yaml` to create your Conda environment:
> ```
> conda env create -f environment.yaml
> conda activate magenta_py38
> ```
> Alternatively, you can install core dependencies with pip using `requirements.txt`, but some features (e.g., GPU support) may require Conda.

> **Requirements:**  
> - Download the [GTZAN dataset](http://marsyas.info/downloads/datasets.html) and place it in the `data/audio` directory.  
> - Download the pre-trained WaveNet model from the project's release page or train your own using the provided scripts. Place the model checkpoint in the root directory.

1.  **Feature Extraction**
 Run preprocessing/feature_extraction.py and preprocessing/feature_preprocessing.py to generate and save MFCCs, WaveNet embeddings and the preprocessed dataset into their respective subdirectories.
2.  **Exploration & Visualization**
In this directory, you can find an example of encoding decoding via wavnet, and feature visualization via TSNE and Umap.
3. **Run Experiments (Python Scripts):**

You can run experiments using with options for architecture and features.To run the experiments, use the following command structure:

```
python main.py --architecture ARCH --features FEAT [other options]
```

**Architectures:**
- `svm`: Support Vector Machine classifier
- `nn_torch`: Feedforward neural network (PyTorch)
- `rnn_torch`: Sequential RNN+NN classifier (PyTorch)

**Features:**
- `base`: Use aggregated features (e.g., from `genrefeatures.csv`)
- `fusion`: Use sequential features (MFCC and WaveNet embeddings)

**SVM Options:**
- `--svm_kernel {linear,poly,rbf,sigmoid}`: Kernel type (default: rbf)
- `--svm_c FLOAT`: Regularization parameter C (default: 1.0)
- `--svm_gamma {scale,auto}` or float: Kernel coefficient (default: scale)

**NN Options:**
- `--nn_epochs INT`: Number of epochs (default: 50)
- `--nn_batch_size INT`: Batch size (default: 32)
- `--nn_lr FLOAT`: Learning rate (default: 0.001)

**RNN+NN Options:**
- `--rnn_epochs INT`: Number of epochs (default: 50)
- `--rnn_batch_size INT`: Batch size (default: 32)
- `--rnn_lr FLOAT`: Learning rate (default: 0.001)

**Other Options:**
- `--wandb_project NAME`: Weights & Biases project name
- `--seed INT`: Random seed

**Example usage:**

Run SVM on aggregated features:
```
python main.py --architecture svm --features base --svm_kernel rbf --svm_c 1.0 --svm_gamma scale
```

Run PyTorch NN on aggregated features:
```
python main.py --architecture nn --features base --nn_epochs 100 --nn_batch_size 16 --nn_lr 0.0005
```

Run RNN+NN on sequential features:
```
python main.py --architecture rnn_nn --features fusion --rnn_epochs 500 --rnn_batch_size 32 --rnn_lr 0.0001 
```

