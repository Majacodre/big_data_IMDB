from utils.fetch_files import fetch_dataset
from utils.merge_files import prepare_imdb_data

# Fetch the dataset files
# fetch_dataset()

# Merge the train files into a single file
prepare_imdb_data(data_folder="data", train_output_csv="data/merged_train_imdb_data.csv", hidden_output_csv="data/merged_hidden_imdb_data.csv", validation_output_csv="data/merged_validation_hidden_imdb_data.csv")