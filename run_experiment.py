"""
run_experiment.py
Main orchestration script for running experiments and logging with wandb.
"""
import wandb
from main import parse_and_validate
# from models import nn, rnn_nn  # Uncomment if these exist
# from models import nn, rnn_nn  # Uncomment if these exist
from models import svm, random_forest
from models import nn_torch, rnn_torch
# from models import nn, rnn_nn  # Uncomment if these exist


def run_experiment_main(args):
    """
    Main orchestration function for running experiments and logging with wandb.
    Args:
        args: Parsed arguments from main.py
    """
    # 2. Initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    # 3. Select and instantiate the model based on args.architecture
    if args.architecture == 'svm':
        model = svm.SVMModel(args)
    elif args.architecture == 'nn':
        model = nn_torch.TorchNNModel(args)
    elif args.architecture == 'rnn_nn':
        model = rnn_torch.TorchRNNExperiment(args)
    # elif args.architecture == 'nn':
    #     model = nn.NNModel(args)
    # elif args.architecture == 'rnn_nn':
    #     model = rnn_nn.RNNNNModel(args)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")


    # 4. Train the model
    if hasattr(model, 'train'):
        model.train()

    # 5. Evaluate the model
    if hasattr(model, 'evaluate'):
        metrics = model.evaluate()
        print("Evaluation metrics:", metrics)

    # 6. Optionally save model/checkpoints
    if hasattr(model, 'save'):
        save_path = getattr(args, 'save_path', None)
        if args.architecture == 'svm':
            save_path = save_path or 'svm_model_kernel{}_C{}_gamma{}.joblib'.format(
                getattr(args, 'svm_kernel', 'rbf'),
                getattr(args, 'svm_c', 1.0),
                getattr(args, 'svm_gamma', 'scale'))
        else:
            save_path = save_path or 'model_checkpoint.pt'
        model.save(save_path)

    # 9. Finish wandb run
    wandb.finish()

    # Note: Implement the actual logic for data loading, training, evaluation, and saving in the respective model classes and utility functions.
