from utils.fetch_files import fetch_dataset
from utils.merge_files import prepare_imdb_data
from utils.cleaning import profile_data, clean_data
from utils.features import build_features
from utils.model_baseline import run as run_model
from utils.merge_rt import merge_with_rotten_tomatoes


# Fetch the dataset files
# fetch_dataset()

# Merge the train files into a single file
# prepare_imdb_data(
#     data_folder="data",
#     train_output_csv="data/merged_train_imdb_data.csv",
#     hidden_output_csv="data/merged_hidden_imdb_data.csv",
#     validation_output_csv="data/merged_validation_hidden_imdb_data.csv"
# )

# Profiling
profile_data("data/merged_train_imdb_data.csv")

# Cleaning + feature building + baseline model

clean_data(
    train_csv="data/merged_train_imdb_data.csv",
    val_csv="data/merged_validation_hidden_imdb_data.csv",
    test_csv="data/merged_hidden_imdb_data.csv",
    train_out="data/clean_train.csv",
    val_out="data/clean_validation.csv",
    test_out="data/clean_test.csv",
)
 

build_features(
    train_csv="data/clean_train.csv",
    val_csv="data/clean_validation.csv",
    test_csv="data/clean_test.csv",
    train_out="data/features_train.csv",
    val_out="data/features_validation.csv",
    test_out="data/features_test.csv",
)

# trains and saves submission files
run_model(
    train_csv="data/features_train.csv",
    val_csv="data/features_validation.csv",
    test_csv="data/features_test.csv",
    val_out="submissions/validation_submission.csv",
    test_out="submissions/test_submission.csv",
)

# add external data + analyze new NAs
merge_with_rotten_tomatoes(
    train_csv="data/features_train.csv",
    val_csv="data/features_validation.csv",
    test_csv="data/features_test.csv",
    rt_csv="data/rotten_tomatoes_movies.csv",
    train_out="data/rt_train.csv",
    val_out="data/rt_validation.csv",
    test_out="data/rt_test.csv",
)