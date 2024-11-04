import pandas as pd

class BaseDecomposition:
    def split(self, data: pd.DataFrame):
        raise NotImplementedError("Subclasses should implement this method.")

    def join(self, data_parts: list):
        raise NotImplementedError("Subclasses should implement this method.")

class TableDecomposition(BaseDecomposition):
    def split(self, data: pd.DataFrame):
        split_tables = [data.iloc[:, :len(data.columns)//2], data.iloc[:, len(data.columns)//2:]]
        return split_tables
    
    def join(self, data_parts: list):
        return pd.concat(data_parts, axis=1)

if __name__ == "__main__":
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9],
        'D': [10, 11, 12]
    })
    decomposer = TableDecomposition()
    split_data = decomposer.split(data)
    print("Split Data:")
    for part in split_data:
        print(part)
    joined_data = decomposer.join(split_data)
    print("\nJoined Data:")
    print(joined_data)

