import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--img_path", type=str, default="", help="Image path")
    parser.add_argument("--train_csv_path", type=str, default="/Users/duyhoang/Documents/Research/VQA/VQA_Vi/data/csv/ViTextVQA_train.csv", help="CSV path training")
    parser.add_argument("--test_csv_path", type=str, default="/Users/duyhoang/Documents/Research/VQA/VQA_Vi/data/csv/ViTextVQA_test.csv", help="CSV path testing")
    parser.add_argument("--dev_csv_path", type=str, default="/Users/duyhoang/Documents/Research/VQA/VQA_Vi/data/csv/ViTextVQA_dev.csv", help="CSV path dev")
    return parser.parse_args()