import duckdb
import os

def prepare_imdb_data(data_folder: str, output_csv: str):
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

    # Save to CSV
    con.execute(f"COPY full_data tO '{output_csv}' (HEADER TRUE)")

    print(f"[INFO] Merged CSV saved to: {output_csv}")