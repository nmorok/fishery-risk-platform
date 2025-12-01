import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np
import os

DB_PATH = 'data/fishery_disasters.db'
CSV_DIR = Path('data/csv')
SQL_DIR = Path('src/api')

def create_tables():

    # Create connection
    conn = sqlite3.connect(DB_PATH)

    # Read and execute SQL file
    with open(f"{SQL_DIR}/create_tables.sql", 'r') as f:
        sql_script = f.read()

    conn.executescript(sql_script)
    conn.commit()
    conn.close()
    print('Database tables created successfully.')


def load_data():
    conn = sqlite3.connect(DB_PATH)
    print('Loading data into database...')

    tables = [
        'disasters',
        'disaster_relationships',
        'disaster_years',
        'species',
        'fishery_events',
        'event_disasters',
        'event_species',
        'event_years',
        'sst_anomalies_metadata'
    ]

    for table in tables:
        csv_file = f"{CSV_DIR}/{table}.csv"
        
        if not os.path.exists(csv_file):
            print(f"CSV file for table '{table}' not found at {csv_file}. Skipping.")
            continue

        df = pd.read_csv(csv_file)

        df = df.replace({pd.NA: None})  # Replace pandas NA with None for SQL compatibility

        df.to_sql(table, conn, if_exists='append', index=False)
        print(f"Loaded data into table '{table}' from '{csv_file}'.")

    conn.commit()

    # quick verification
    cursor = conn.cursor()
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' has {count} records.")

    conn.close()
    print('Data loading complete.')

def load_specific_table(table_name):
    conn = sqlite3.connect(DB_PATH)
    print(f'Loading data into table {table_name}...')

    csv_file = f"{CSV_DIR}/{table_name}.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"CSV file for table '{table_name}' not found at {csv_file}.")
        return
    

    df = pd.read_csv(csv_file)

    df = df.replace({pd.NA: None})  # Replace pandas NA with None for SQL compatibility

    df.to_sql(table_name, conn, if_exists='append', index=False)
    print(f"Loaded data into table '{table_name}' from '{csv_file}'.")

    conn.commit()

    # quick verification
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Table '{table_name}' has {count} records.")

    conn.close()
    print(f'Data loading for table {table_name} complete.')

def print_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:")
    for table in tables:
        print(table[0])

    conn.close()

def print_table_contents(table_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()

    print(f"Contents of table '{table_name}':")
    for row in rows:
        print(row)

    conn.close()

def create_csv():
    output_file = 'data/csv/model_data.csv'
    
    conn = sqlite3.connect(DB_PATH)
    query = '''
    SELECT 
        -- Event identifiers
        fe.event_id,
        fe.fishery_name,
        fe.management_zone,
        fe.sst_region,
        
        -- Event characteristics
        fe.num_fisheries,
        fe.num_species,
        fe.num_years,
        fe.total_value,
        fe.appropriation,
        fe.sst_year,
        
        -- Heatwave metrics
        hm.region,
        hm.peak_intensity,
        hm.duration_days,
        hm.mean_duration,
        hm.median_duration,
        hm.sum_cumulative_intensity,
        hm.mean_cumulative_intensity,
        hm.median_cumulative_intensity,
        hm.percent_in_heatwave,
        hm.total_cells,
        hm.cells_with_heatwave,
        hm.total_events,
        
        -- Aggregated disaster info
        GROUP_CONCAT(DISTINCT ed.disaster_id) as disaster_ids,
        COUNT(DISTINCT ed.disaster_id) as num_disasters,
        
        -- Aggregated species info
        GROUP_CONCAT(DISTINCT s.species_name) as species_names,
        GROUP_CONCAT(DISTINCT s.species_family) as species_families,
        
        -- Year info
        MIN(ey.year) as first_year,
        MAX(ey.year) as last_year,
        (SELECT GROUP_CONCAT(year, ',') 
        FROM (SELECT DISTINCT year FROM event_years WHERE event_id = fe.event_id ORDER BY year)
        ) as years
        
    FROM fishery_events fe
    LEFT JOIN heatwave_metrics hm ON fe.event_id = hm.event_id
    LEFT JOIN event_disasters ed ON fe.event_id = ed.event_id
    LEFT JOIN event_species es ON fe.event_id = es.event_id
    LEFT JOIN species s ON es.species_id = s.species_id
    LEFT JOIN event_years ey ON fe.event_id = ey.event_id
    
    GROUP BY fe.event_id
    ORDER BY fe.event_id
    '''

    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Ensure concatenated fields are strings (not interpreted as numbers)
    string_columns = ['disaster_ids', 'species_names', 'species_families', 'years']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Create categorical variables for modeling
    df['zone_category'] = pd.Categorical(df['management_zone'])
    df['species_category'] = pd.Categorical(df['species_families'])
    df['year_category'] = pd.Categorical(df['sst_year'])
    
    # Log transform economic variables (common for $ amounts)
    df['log_appropriation'] = np.log(df['appropriation'] + 1)  # +1 to handle zeros
    df['log_total_value'] = np.log(df['total_value'] + 1)
    
    # Calculate derived metrics (handle division by zero)
    df['percent_cells_in_heatwave'] = np.where(
        df['total_cells'] > 0,
        (df['cells_with_heatwave'] / df['total_cells']) * 100,
        0
    )
    df['avg_intensity_per_event'] = np.where(
        df['total_events'] > 0,
        df['sum_cumulative_intensity'] / df['total_events'],
        0
    )
    df['avg_duration_per_event'] = np.where(
        df['total_events'] > 0,
        df['duration_days'] / df['total_events'],
        0
    )
    
    df.to_csv(output_file, index=False)
    print(f'Model data CSV created at {output_file}.')
    print(f'Total rows: {len(df)}')
    print(f'Total columns: {len(df.columns)}')
    
    # Show sample of years column
    print(f'\nSample of years column:')
    print(df[['event_id', 'years']].head(10))


def delete_table(table_name):
    """Delete a table from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"✓ Table '{table_name}' deleted successfully.")
    except sqlite3.Error as e:
        print(f"✗ Error deleting table '{table_name}': {e}")
    finally:
        conn.close()    

def create_specific_table(table_name):
    """Create a specific table from the SQL schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Read the SQL file
        with open(f"{SQL_DIR}/create_tables.sql", 'r') as f:
            sql_script = f.read()
        
        # Split by semicolons to get individual statements
        statements = sql_script.split(';')
        
        # Find the CREATE TABLE statement for this specific table
        table_statement = None
        for statement in statements:
            # Check if this statement contains CREATE TABLE for our table
            if f"CREATE TABLE" in statement.upper() and table_name.lower() in statement.lower():
                table_statement = statement.strip() + ';'
                break
        
        if table_statement:
            cursor.execute(table_statement)
            conn.commit()
            print(f"✓ Table '{table_name}' created successfully.")
        else:
            print(f"✗ CREATE TABLE statement for '{table_name}' not found in SQL file.")
            
    except sqlite3.Error as e:
        print(f"✗ Error creating table '{table_name}': {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    #create_tables() # to create tables
    #load_data()  # to load all tables
    #load_specific_table('{table_name}')  # to load a specific table
    #load_specific_table('sst_anomalies_metadata')
    create_csv()  # to create csv files from database
    #delete_table('heatwave_metrics')  # to delete a specific table
    #create_specific_table('heatwave_metrics')  # to create a specific table
    print_tables()  # to print tables in the database
    print_table_contents('heatwave_metrics')  # to print contents of a specific table
