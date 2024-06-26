import clickhouse_connect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connection details
host = 'localhost'
port = '8123'
user = 'default'
password = ''
database = 'default'
table = 'cemex'
new_table = 'cemex_sample'


# Function to describe the table
def describe_table_rows(client, table):
        schema_query = f"DESCRIBE TABLE {table}"
        schema = client.query(schema_query)
        schema_data = list(schema.result_rows)
        return schema_data

# Function to describe the table
def describe_table_colums(client, table):
        schema_query = f"DESCRIBE TABLE {table}"
        schema = client.query(schema_query)
        schema_columns = [col[0] for col in schema.result_columns]
        return schema_columns

# Function to calculate statistics for a column
def get_column_statistics(client, table, column, dtype):
    
    if 'Array(Nullable(String))' in dtype:
        stats_query = f"""
        SELECT 
            arrayJoin({column}) AS element, 
            count(*) AS frequency 
        FROM {table} 
        GROUP BY element
        """
    
    elif 'Int' in dtype or 'Float' in dtype:
        stats_query = f"""
        SELECT 
            avg({column}) AS mean, 
            stddevPop({column}) AS stddev, 
            min({column}) AS min, 
            max({column}) AS max 
        FROM {table}
        """
    elif 'String' in dtype or 'Boolean' in dtype:
        stats_query = f"""
        SELECT 
            {column}, 
            count(*) AS frequency 
        FROM {table} 
        GROUP BY {column}
        """
    elif 'DateTime64' in dtype:
        stats_query = f"""
        SELECT 
            min({column}) AS min, 
            max({column}) AS max 
        FROM {table}
        """
    
    else:
        stats_query = None

    if stats_query:
        stats = client.query(stats_query)
        return stats

# Function to generate synthetic data based on statistics
def generate_synthetic_data(statistics, num_rows):
    synthetic_data = {}

    for column, stats in statistics.items():
        if isinstance(stats, str):
            print(f"No statistics available for column: {column}")
            continue

        dtype = schema_df[schema_df['name'] == column]['type'].values[0]

        if 'Array(Nullable(String))' in dtype:
            # For Array(String) data
            elements, frequencies = zip(*stats)
            probabilities = np.array(frequencies) / sum(frequencies)
            array_lengths = np.random.randint(1, 10, size=num_rows)  # Random array lengths between 1 and 10
            synthetic_data[column] = [
                list(np.random.choice(elements, size=length, p=probabilities))
                for length in array_lengths
            ]
        elif 'String' in dtype or 'Boolean' in dtype:
            # For categorical data or Boolean
            categories, frequencies = zip(*stats)
            probabilities = np.array(frequencies) / sum(frequencies)
            synthetic_data[column] = np.random.choice(categories, size=num_rows, p=probabilities)
        elif 'DateTime64' in dtype:
            # For DateTime64 data
            stats_df = pd.DataFrame(stats, columns=['min', 'max'])
            min_val = pd.to_datetime(stats_df['min'].values[0])
            max_val = pd.to_datetime(stats_df['max'].values[0])
            synthetic_data[column] = [min_val + (max_val - min_val) * np.random.rand() for _ in range(num_rows)]            
        else:
            # For numerical data
            stats_df = pd.DataFrame(stats, columns=['mean', 'stddev', 'min', 'max'])
            mean = stats_df['mean'].values[0]
            stddev = stats_df['stddev'].values[0]
            min_val = stats_df['min'].values[0]
            max_val = stats_df['max'].values[0]
            synthetic_data[column] = np.random.normal(loc=mean, scale=stddev, size=num_rows)
            synthetic_data[column] = np.clip(synthetic_data[column], min_val, max_val)
            if 'Int' in dtype:
                synthetic_data[column] = synthetic_data[column].astype(int)

    return pd.DataFrame(synthetic_data)


# Function to insert data into ClickHouse
def insert_data(client, table, df):
    ##columns = ', '.join(df.columns)
    
    columns = [col for col in df.columns]    
    values = [tuple(row) for row in df.to_numpy()]
    query = f"INSERT INTO {table} ({columns}) VALUES"
    ##client.execute(query, values)
    client.insert(table, values, columns)



# Connect to ClickHouse
##client = clickhouse_connect.get_client( connect_timeout=200, query_limit=0,compress=False, host=host, port=port, user=user, password=password,  database=database)
client = clickhouse_connect.get_client( query_limit=0,compress='lz4', host=host, port=port, user=user, password=password,  database=database)

# Retrieve schema
schema_data = describe_table_rows(client, table)
schema_columns = describe_table_colums(client, table)
   
# Create DataFrame from schema data
schema_df = pd.DataFrame(schema_data, columns=['name', 'type', '','','','',''])
print (schema_df);


# Generate statistics for each column
statistics = {}
for index, row in schema_df.iterrows():
    column_name = row['name']
    column_type = row['type']
    print(f"Processing column: {column_name} ({column_type})")
    column_stats = get_column_statistics(client, table, column_name, column_type)
    statistics[column_name] = column_stats.result_rows if column_stats else "No statistics available"

# Number of rows to generate
num_rows = 100000
total_rows = 100000000
iteration = int(total_rows/num_rows)

for i in range( iteration):
    timestamp_start = datetime.now()
    # Generate synthetic data
    synthetic_data_df = generate_synthetic_data(statistics, num_rows)

    # Display first few rows of synthetic data
    print(synthetic_data_df.head())

    # Insert synthetic data into ClickHouse table
    insert_data(client, new_table, synthetic_data_df)

    timestamp_end = datetime.now()
    print(f"Elapsed time for row {num_rows}: {timestamp_end - timestamp_start}")
    print(f"Estimated remaining time: {(total_rows - i*num_rows)/num_rows * (timestamp_end - timestamp_start)}")


# Close the client
client.close()