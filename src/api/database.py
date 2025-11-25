import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np

DB_PATH = 'data/fishery_disasters.db'
CSV_DIR = Path('data/csv')

def create_tables():
    # Create connection
    conn = sqlite3.connect(DB_PATH)

    # Read and execute SQL file
    with open(f"{CSV_DIR}/create_tables.sql", 'r') as f:
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
        'event_years'
    ]

    for table in tables:
        csv_file = f"{CSV_DIR}/{table}.csv"
        
        if not csv_file.exists():
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
    
    if not csv_file.exists():
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
    print(f'Data loading for table {table_name} complete.')    .

def create_csv():
    output_file = 'model_data.csv'
    
    conn = sqlite3.connect(DB_PATH)
    query = '''
    SELECT 
        -- Event identifiers
        fe.event_id, -- int
        fe.fishery_name, -- text
        fe.management_zone, -- text
        fe.sst_region, -- text
        
        -- Event characteristics
        fe.num_fisheries, -- int
        fe.num_species, -- int
        fe.num_years, -- int
        fe.total_value, -- float
        fe.appropriation, -- float
        fe.sst_year, -- int
        
        -- Heatwave metrics
        hm.max_sst_anomaly, -- float
        hm.duration_days, -- int
        hm.cumulative_intensity, -- float
        hm.mean_intensity, -- float
        hm.percent_in_heatwave, -- float
        
        -- Aggregated disaster info
        GROUP_CONCAT(DISTINCT ed.disaster_id) as disaster_ids, -- text
        COUNT(DISTINCT ed.disaster_id) as num_disasters, -- int
        
        -- Aggregated species info
        GROUP_CONCAT(DISTINCT s.species_name) as species_names, -- text
        GROUP_CONCAT(DISTINCT s.species_family) as species_families, -- text
        
        -- Year info
        MIN(ey.year) as first_year,
        MAX(ey.year) as last_year,
        GROUP_CONCAT(DISTINCT ey.year ORDER BY ey.year) as years
        
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

    # clean df
 
    
    # Create categorical variables for modeling
    df['zone_category'] = pd.Categorical(df['management_zone'])
    df['species_category'] = pd.Categorical(df['species_families'])
    df['year'] = pd.Categorical(df['sst_year'])

    
    # Log transform economic variables (common for $ amounts)
    df['log_appropriation'] = np.log(df['appropriation'] + 1)  # +1 to handle zeros
    df['log_total_value'] = np.log(df['total_value'] + 1)
    
    df.to_csv(output_file, index=False)
    print(f'Model data CSV created at {output_file}.')


if __name__ == "__main__":
    create_tables() # to create tables
    load_data()  # to load all tables
    # load_specific_table('{table_name}')  # to load a specific table
    create_csv()  # to create csv files from database