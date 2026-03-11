import duckdb
import os

def prepare_imdb_data(data_folder: str, train_output_csv: str, hidden_output_csv: str, validation_output_csv: str):
    """
    Load IMDB train CSVs and writing/directing JSONs, merge them, 
    and save a single CSV for analysis.

    Args:
        data_folder (str): Folder containing all input files
        output_csv (str): Path to save the merged CSV
    """
    # Create an in-memory DuckDB connection
    con = duckdb.connect(database=':memory:')

    # Merge all train CSVs
    csv_pattern = os.path.join(data_folder, "train-*.csv")
    con.execute(f"""
        CREATE TABLE train AS
        SELECT * FROM read_csv_auto('{csv_pattern}')
    """)

    # Load writing JSON
    con.execute(f"""
        CREATE TABLE writing AS
        SELECT * FROM read_json_auto('{data_folder}/writing.json')
    """)

    # Load directing JSON
    con.execute(f"""
        CREATE TABLE directing AS
        SELECT * FROM read_json_auto('{data_folder}/directing.json')
    """)

    con.execute(f"""
        CREATE TABLE hidden_data AS
        SELECT * FROM read_csv_auto('{data_folder}/test_hidden.csv')
    """)

    con.execute(f"""
        CREATE TABLE validation_hidden AS
        SELECT *
        FROM read_csv_auto('{data_folder}/validation_hidden.csv')
    """)

    # Group writers by movie
    con.execute("""
        CREATE TABLE writers_grouped AS
        SELECT CAST(movie AS VARCHAR) AS tconst, string_agg(writer, ',') AS writers
        FROM writing
        GROUP BY movie
    """)

    # Group directors by movie
    con.execute("""
        CREATE TABLE directors_grouped AS
        SELECT CAST(movie AS VARCHAR) AS tconst, string_agg(director, ',') AS directors
        FROM directing
        GROUP BY movie
    """)

    con.execute("""
        CREATE TABLE full_data AS
        SELECT t.*, w.writers, d.directors
        FROM train t
        LEFT JOIN writers_grouped w USING(tconst)
        LEFT JOIN directors_grouped d USING(tconst)
    """)


  
    con.execute(""" 
        CREATE TABLE test_hidden_merged AS
        SELECT *
        FROM hidden_data
        LEFT JOIN writers_grouped w USING(tconst)
        LEFT JOIN directors_grouped d USING(tconst)
    """)

    con.execute(""" 
        CREATE TABLE validation_hidden_merged AS
        SELECT *
        FROM validation_hidden
        LEFT JOIN writers_grouped w USING(tconst)
        LEFT JOIN directors_grouped d USING(tconst)
    """)
    # Save to CSV
    con.execute(f"COPY full_data tO '{train_output_csv}' (HEADER TRUE)")
    con.execute(f"COPY hidden_data tO '{hidden_output_csv}' (HEADER TRUE)")
    con.execute(f"COPY validation_hidden tO '{validation_output_csv}' (HEADER TRUE)")


    print(f"[INFO] Merged CSV saved to: {train_output_csv}")
    print(f"[INFO] Merged hidden CSV saved to: {hidden_output_csv}")
    print(f"[INFO] Merged validation hidden CSV saved to: {validation_output_csv}")