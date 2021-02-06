from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser(description='Create a model folder for the fine-tuned model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--create', type=str, default='mymodel',
                    help='Name of the model folder to create')
parser.add_argument('-m', '--model', type=str, default='124M',
                    help='Name of the GPT-2 model used. Ex: 124M')
parser.add_argument('-r', '--run', type=str, default='run1', help='Name of the run. Ex: run2')
args = parser.parse_args()


def main():
    custom_model_name = args.create
    model_name = args.model
    run_name = args.run

    # Models folder
    models_folder = Path('../models')

    # Create new folder for the model name inputted by user
    Path.mkdir(models_folder.joinpath(custom_model_name))
    custom_model_folder = models_folder.joinpath(custom_model_name)

    # GPT-2 Model folder
    model_folder = Path(models_folder.joinpath(model_name))

    # Copy files from gpt-2 model to user inputted model folder
    # copy encoder.json
    shutil.copy(model_folder.joinpath('encoder.json'), custom_model_folder)
    # copy hparams.json
    shutil.copy(model_folder.joinpath('hparams.json'), custom_model_folder)
    # copy vocab.bpe
    shutil.copy(model_folder.joinpath('vocab.bpe'), custom_model_folder)

    # User inputted run folder inside the checkpoint folder
    run_folder = Path('./checkpoint').joinpath(run_name)

    # Copy files from run folder to user inputted model folder
    # copy model files (3 files)
    for file in run_folder.glob('model*'):
        shutil.copy(file, custom_model_folder)

    # Copy checkpoint file to user inputted model folder
    shutil.copy(run_folder.joinpath('checkpoint'), custom_model_folder)


if __name__ == '__main__':
    main()
