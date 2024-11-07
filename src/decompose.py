import pandas as pd
from sqlalchemy import create_engine, inspect
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from collections import defaultdict
from tqdm import tqdm

class TableNormalizer:
    def __init__(self, df):
        """
        Initialize normalizer with a pandas DataFrame
        """
        self.df = df
        self.dependencies = defaultdict(list)
        
    def identify_functional_dependencies(self, threshold=0.95):
        """
        Identify potential functional dependencies between columns
        """
        columns = self.df.columns
        for col1 in tqdm(columns):
            for col2 in tqdm(columns):
                if col1 != col2:
                    # Check if col1 determines col2
                    grouped = self.df.groupby(col1)[col2].nunique()
                    if (grouped == 1).all():
                        self.dependencies[col1].append(col2)
                        
    def suggest_decomposition(self):
        """
        Suggest table decomposition based on identified dependencies
        """
        tables = []
        processed_cols = set()
        
        # Find candidate keys
        for col, dependents in self.dependencies.items():
            if col not in processed_cols:
                # Create new table with determinant and its dependents
                table_cols = [col] + dependents
                new_table = self.df[table_cols].drop_duplicates()
                tables.append({
                    'name': f'table_{len(tables)+1}',
                    'columns': table_cols,
                    'data': new_table
                })
                processed_cols.update(table_cols)
        
        # Create table for remaining columns
        remaining_cols = [col for col in self.df.columns if col not in processed_cols]
        if remaining_cols:
            tables.append({
                'name': f'table_{len(tables)+1}',
                'columns': remaining_cols,
                'data': self.df[remaining_cols].drop_duplicates()
            })
            
        return tables

    def export_to_sql(self, tables, engine):
        """
        Export normalized tables to SQL database
        """
        for table in tables:
            table['data'].to_sql(table['name'], engine, index=False, if_exists='replace')

# Example usage
def normalize_example():
    # Create sample data
    # data = {
    #     'OrderID': [1, 1, 2, 3],
    #     'CustomerName': ['John', 'John', 'Mary', 'Bob'],
    #     'CustomerEmail': ['john@email.com', 'john@email.com', 'mary@email.com', 'bob@email.com'],
    #     'Product': ['Laptop', 'Mouse', 'Phone', 'Tablet'],
    #     'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics'],
    #     'Price': [1000, 25, 800, 500]
    # }
    from sdv.datasets.demo import download_demo
    
    real_data, metadata = download_demo(modality='single_table', dataset_name='covtype')
    
    df = pd.DataFrame(real_data)
    
    print("Original data:")
    print(df)
    
    # Initialize normalizer
    normalizer = TableNormalizer(df)
    
    # Identify dependencies
    normalizer.identify_functional_dependencies()
    
    # Get suggested decomposition
    normalized_tables = normalizer.suggest_decomposition()
    
    # Create SQLite database for demonstration
    engine = create_engine('sqlite:///normalized_database.db')
    
    # Export to SQL
    normalizer.export_to_sql(normalized_tables, engine)
    
    return normalized_tables

# Run the example
tables = normalize_example()

# Print the results
for table in tables:
    print(f"\nTable: {table['name']}")
    print("Columns:", table['columns'])
    print(table['data'])