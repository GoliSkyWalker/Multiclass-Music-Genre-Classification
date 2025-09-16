# Entry point for the music feature extraction and classification pipeline
import argparse
def argument_setup():

    
    parser = argparse.ArgumentParser(description="Set up parameters for the experiment")

    # Experiment definition
    parser.add_argument('--architecture', type=str, required=True, choices=['svm', 'nn', 'rnn_nn'],
                        help="Model architecture: 'svm', 'nn', or 'rnn_nn'.")
    parser.add_argument('--features', type=str, required=True, choices=['base', 'fusion'],
                        help="Type of features to use: 'base' or 'fusion'.")


    # Arguments for NN experiments
    parser.add_argument('--nn_epochs', type=int, default=50,
                        help="Number of training epochs for NN.")
    parser.add_argument('--nn_batch_size', type=int, default=32,
                        help="Training batch size for NN.")
    parser.add_argument('--nn_lr', type=float, default=0.001,
                        help="Learning rate for NN.")

    # Arguments for RNN+NN experiments
    parser.add_argument('--rnn_epochs', type=int, default=50,
                        help="Number of training epochs for RNN+NN.")
    parser.add_argument('--rnn_batch_size', type=int, default=32,
                        help="Training batch size for RNN+NN.")
    parser.add_argument('--rnn_lr', type=float, default=0.001,
                        help="Learning rate for RNN+NN.")

    # Arguments for SVM experiments
    parser.add_argument('--svm_kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="SVM kernel type.")
    parser.add_argument('--svm_c', type=float, default=1.0,
                        help="SVM regularization parameter C.")
    parser.add_argument('--svm_gamma', type=str, default='scale',
                        help="SVM kernel coefficient gamma (float or 'scale'/'auto').")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")

    # Wandb Configuration
    parser.add_argument('--wandb_project', type=str, default="MyExperiments",
                        help="Weights & Biases project name.")

    args = parser.parse_args()
    return args , parser
def validate_arguemnts(args,parser):
    """
    Validate that only relevant arguments are used for each architecture.
    """
    arch = args.architecture
    errors = []
    # NN
    if arch == 'nn':
        # SVM and RNN+NN args should not be set (if user sets them explicitly)
        if args.svm_kernel != 'rbf' or args.svm_c != 1.0 or args.svm_gamma != 'scale':
            errors.append("SVM arguments are not valid for NN architecture.")
        if args.rnn_epochs != 50 or args.rnn_batch_size != 32 or args.rnn_lr != 0.001:
            errors.append("RNN+NN arguments are not valid for NN architecture.")
    # SVM
    elif arch == 'svm':
        # NN and RNN+NN args should not be set
        if args.nn_epochs != 50 or args.nn_batch_size != 32 or args.nn_lr != 0.001:
            errors.append("NN arguments are not valid for SVM architecture.")
        if args.rnn_epochs != 50 or args.rnn_batch_size != 32 or args.rnn_lr != 0.001:
            errors.append("RNN+NN arguments are not valid for SVM architecture.")
    # RNN+NN
    elif arch == 'rnn_nn':
        # NN and SVM args should not be set
        if args.nn_epochs != 50 or args.nn_batch_size != 32 or args.nn_lr != 0.001:
            errors.append("NN arguments are not valid for RNN+NN architecture.")
        if args.svm_kernel != 'rbf' or args.svm_c != 1.0 or args.svm_gamma != 'scale':
            errors.append("SVM arguments are not valid for RNN+NN architecture.")
    if errors:
        for err in errors:
            print(f"Argument Error: {err}")
        parser.print_help()
        exit(1)


def parse_and_validate():
    args, parser = argument_setup()
    try:
        validate_arguemnts(args, parser)
    except SystemExit:
        print("Try again with valid arguments")
    return args

if __name__ == '__main__':
    args, parser = argument_setup()
    try:
        validate_arguemnts(args, parser)
    except SystemExit:
        print("Try again with valid arguments")
        exit(1)

    # Directly call the experiment runner
    from run_experiment import run_experiment_main
    run_experiment_main(args)
