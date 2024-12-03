import pandas as pd
import numpy as np
from collections import defaultdict, deque
from scipy.stats import norm
import time
import pandas as pd
import itertools
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

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

class BayesianDecomposition(BaseDecomposition):
    def compute_mutual_information(self, df, var_x, var_y):
        # Compute mutual information between two discrete variables
        joint_counts = df.groupby([var_x, var_y]).size().reset_index(name='count')
        joint_total = joint_counts['count'].sum()
        joint_probs = joint_counts.copy()
        joint_probs['prob'] = joint_probs['count'] / joint_total

        px = df[var_x].value_counts(normalize=True)
        py = df[var_y].value_counts(normalize=True)

        mi = 0.0
        for idx, row in joint_probs.iterrows():
            p_xy = row['prob']
            p_x = px[row[var_x]]
            p_y = py[row[var_y]]
            mi += p_xy * np.log(p_xy / (p_x * p_y) + 1e-9)
        return mi

    def find(self, parent, u):
        while parent[u] != u:
            u = parent[u]
        return u

    def union(self, parent, u, v):
        parent_u = self.find(parent, u)
        parent_v = self.find(parent, v)
        if parent_u != parent_v:
            parent[parent_v] = parent_u
            return True
        else:
            return False

    def get_cpt(self, df):
        variables = list(df.columns)

        # Compute mutual information matrix
        num_variables = len(variables)
        mi_matrix = np.zeros((num_variables, num_variables))
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                mi = self.compute_mutual_information(df, variables[i], variables[j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        edges = []
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                edges.append((variables[i], variables[j], mi_matrix[i, j]))

        # Sort edges by weight (mutual information) in decreasing order
        edges_sorted = sorted(edges, key=lambda x: -x[2])

        # Union-Find data structure for Kruskal's algorithm
        parent = {var: var for var in variables}

        # Kruskal's algorithm to find MST
        mst = []
        for u, v, w in edges_sorted:
            if self.union(parent, u, v):
                mst.append((u, v))
            if len(mst) == num_variables - 1:
                break

        # Build adjacency list from MST
        adj = defaultdict(list)
        for u, v in mst:
            adj[u].append(v)
            adj[v].append(u)

        # Choose root (e.g., first variable)
        root = variables[0]
        visited = set()
        parent_map = {}

        queue = deque()
        queue.append(root)
        visited.add(root)
        parent_map[root] = None

        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)

        cpts = {}
        for var in variables:
            parent = parent_map[var]
            if parent is None:
                # Root node, compute marginal distribution
                counts = df[var].value_counts(normalize=True).reset_index()
                counts.columns = [var, 'probability']
                cpts[var] = counts
            else:
                # Compute conditional probability table
                counts = df.groupby([parent, var]).size().reset_index(name='count')
                total_counts = df.groupby(parent).size().reset_index(name='total_count')
                merged = counts.merge(total_counts, on=parent)
                merged['probability'] = merged['count'] / merged['total_count']
                cpts[var] = merged[[parent, var, 'probability']]
        return cpts, parent_map
    
    # Add Gaussian approximation for conditional sampling of continuous variables
    def sample_from_bn(self, df, cpts, variables, parent_map, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = {}
            for var in variables:
                parent = parent_map[var]
                if parent is None:
                    # Sample from marginal distribution
                    if var in cpts:
                        probs = cpts[var]
                        values = probs[var].values
                        probabilities = probs['probability'].values
                        sampled_value = np.random.choice(values, p=probabilities)
                    else:
                        sampled_value = np.random.normal(df[var].mean(), df[var].std())
                else:
                    # Sample from conditional distribution
                    parent_value = sample.get(parent)
                    if parent_value is not None:
                        cpt = cpts.get(var)
                        if cpt is not None and parent in cpt.columns:
                            cpt_filtered = cpt[cpt[parent] == parent_value]
                            if not cpt_filtered.empty:
                                values = cpt_filtered[var].values
                                probabilities = cpt_filtered['probability'].values
                                sampled_value = np.random.choice(values, p=probabilities)
                            else:
                                sampled_value = np.random.normal(df[var].mean(), df[var].std())
                        else:
                            sampled_value = np.random.normal(df[var].mean(), df[var].std())
                    else:
                        sampled_value = np.random.normal(df[var].mean(), df[var].std())
                # Quantize based on original data type
                if np.issubdtype(df[var].dtype, np.integer):
                    sample[var] = int(round(sampled_value))
                elif np.issubdtype(df[var].dtype, np.floating):
                    sample[var] = float(sampled_value)
                else:
                    sample[var] = sampled_value  # for any non-numeric columns (if any)
            samples.append(sample)
        return pd.DataFrame(samples)

    def inference_generation(self, df, cpts, parent_map, num_samples, verbose=False):
        # Generate sampled data
        variables = list(df.columns)
        sampled_data = self.sample_from_bn(df, cpts, variables, parent_map, num_samples=num_samples)
        if verbose:
            # Compare marginal distributions
            for var in variables:
                original_dist = df[var].value_counts(normalize=True)
                sampled_dist = sampled_data[var].value_counts(normalize=True)
                comparison = pd.DataFrame({'Original': original_dist, 'Sampled': sampled_dist})
                print(f'Distribution for {var}:')
                print(comparison)
                print('\n')
        return sampled_data

    def split(self, data: pd.DataFrame):
        self.cpts, self.parent_map = self.get_cpt(data)
    
    def join(self, data, num_samples):
        return self.inference_generation(data, self.cpts, self.parent_map, num_samples)

class NFDecomposition(BaseDecomposition):
    def __init__(self, data):
        self.data = data
        self.tables = [data]
        self.functional_dependencies = self._find_initial_fds()
    
    def _find_initial_fds(self):
        """Finds initial functional dependencies in the data by analyzing uniqueness."""
        columns = self.data.columns
        fds = []
        
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    grouped = self.data.groupby(col1)[col2].nunique()
                    if all(grouped == 1):  # Col2 is functionally dependent on col1
                        fds.append((col1, col2))
        return fds
    
    def to_1nf(self):
        """Ensures the table is in 1NF by removing any non-atomic values."""
        for col in self.data.columns:
            if any(isinstance(i, list) for i in self.data[col]):
                self.data = self.data.explode(col)
        self.tables[0] = self.data  # Update the main table to be in 1NF
        return self.data
    
    def to_2nf(self):
        """Decomposes the table to 2NF by removing partial dependencies."""
        candidate_keys = self._find_candidate_keys()
        
        # Find partial dependencies for each candidate key
        partial_dependencies = []
        for key in candidate_keys:
            non_key_columns = [col for col in self.data.columns if col not in key]
            for col in non_key_columns:
                for sub_key in key:
                    if (sub_key, col) in self.functional_dependencies:
                        partial_dependencies.append((sub_key, col))
        
        tables_2nf = self._decompose_by_dependencies(partial_dependencies)
        self.tables = self._remove_redundant_tables(tables_2nf)
        return self.tables

    def to_3nf(self):
        """Decomposes the table to 3NF by removing transitive dependencies."""
        minimal_cover = self._find_minimal_cover()
        transitive_dependencies = []
        
        for lhs, rhs in minimal_cover:
            if lhs != rhs and any(rhs != col and (col, rhs) in minimal_cover for col in self.data.columns):
                transitive_dependencies.append((lhs, rhs))
        
        tables_3nf = self._decompose_by_dependencies(transitive_dependencies)
        self.tables = self._remove_redundant_tables(tables_3nf)
        return self.tables
    
    def _find_candidate_keys(self):
        """Identify candidate keys by finding minimal sets of columns that uniquely identify rows."""
        columns = self.data.columns
        candidate_keys = []
        
        for i in range(1, len(columns) + 1):
            for combination in itertools.combinations(columns, i):
                if self.data.duplicated(subset=combination).sum() == 0:  # Unique combination
                    candidate_keys.append(combination)
        return candidate_keys
    
    def _find_minimal_cover(self):
        """Find the minimal cover of functional dependencies."""
        minimal_cover = self.functional_dependencies.copy()
        for fd in self.functional_dependencies:
            if self._is_redundant(fd, minimal_cover):
                minimal_cover.remove(fd)
        return minimal_cover

    def _is_redundant(self, fd, fd_set):
        """Check if a functional dependency is redundant in the FD set."""
        lhs, rhs = fd
        reduced_fd_set = [f for f in fd_set if f != fd]
        closure = self._closure(set([lhs]), reduced_fd_set)
        return rhs in closure

    def _closure(self, attributes, fds):
        """Compute the closure of a set of attributes under given functional dependencies."""
        closure = set(attributes)
        added = True
        while added:
            added = False
            for lhs, rhs in fds:
                if set(lhs).issubset(closure) and rhs not in closure:
                    closure.add(rhs)
                    added = True
        return closure

    def _decompose_by_dependencies(self, dependencies):
        """Decomposes the table by splitting it based on functional dependencies."""
        tables = []
        groups = defaultdict(list)
        
        # Group by the left-hand side of the dependencies
        for lhs, rhs in dependencies:
            groups[lhs].append(rhs)
        
        # Create separate tables based on groups
        for lhs, rhs_list in groups.items():
            columns = [lhs] + rhs_list
            # Remove duplicate columns
            columns = list(dict.fromkeys(columns))
            table = self.data[columns].drop_duplicates()
            tables.append(table)
        
        # Add remaining columns as the main table
        all_dependent_cols = set([item for sublist in groups.values() for item in sublist])
        remaining_columns = [col for col in self.data.columns if col not in all_dependent_cols]
        if remaining_columns:
            main_table = self.data[remaining_columns].drop_duplicates()
            tables.append(main_table)
        
        return tables

    def _remove_redundant_tables(self, tables):
        """Remove tables that are completely contained within other tables."""
        non_redundant_tables = []
        seen_column_sets = set()
        # Sort tables by number of columns (descending) to prefer larger tables
        tables = sorted(tables, key=lambda x: len(x.columns), reverse=True)
        for table in tables:
            # Convert columns to frozenset for hashable comparison
            columns = frozenset(table.columns)
            # Check if this combination of columns is new
            if columns not in seen_column_sets:
                # Check if the table's data is unique
                table_hash = hash(str(table.values.tolist()))
                is_unique = True
                for existing_table in non_redundant_tables:
                    if (table.columns.tolist() == existing_table.columns.tolist() and 
                        table.values.tolist() == existing_table.values.tolist()):
                        is_unique = False
                        break
                if is_unique:
                    seen_column_sets.add(columns)
                    non_redundant_tables.append(table)
        return non_redundant_tables

class TruncateDecomposition(BaseDecomposition):
    def __init__(self, row_fraction: float, col_fraction: float):
        """
        Initialize with row and column fractions.
        :param row_fraction: Fraction for row partitioning (e.g., 0.5 for half, resulting in 2 parts).
        :param col_fraction: Fraction for column partitioning (e.g., 0.5 for half, resulting in 2 parts).
        """
        self.row_fraction = row_fraction
        self.col_fraction = col_fraction

    def split(self, data: pd.DataFrame):
        """
        Split the DataFrame into four partitions:
        - First half of rows with first half of columns
        - First half of rows with second half of columns
        - Second half of rows with first half of columns
        - Second half of rows with second half of columns
        """
        num_rows, num_columns = data.shape
        row_midpoint = int(num_rows * self.row_fraction)
        col_midpoint = int(num_columns * self.col_fraction)

        # Create 4 parts based on row and column midpoints
        part1 = data.iloc[:row_midpoint, :col_midpoint]    # Top-left
        part2 = data.iloc[:row_midpoint, col_midpoint:]    # Top-right
        part3 = data.iloc[row_midpoint:, :col_midpoint]    # Bottom-left
        part4 = data.iloc[row_midpoint:, col_midpoint:]    # Bottom-right

        return [part1, part2, part3, part4]

    def join(self, data_parts: list):
        """
        Recombine the four pieces back into the original DataFrame structure.
        """
        # Concatenate columns for each row part
        top = pd.concat([data_parts[0], data_parts[1]], axis=1)
        bottom = pd.concat([data_parts[2], data_parts[3]], axis=1)
        
        # Concatenate rows to restore original structure
        return pd.concat([top, bottom], axis=0)

class NMFDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        """
        Initialize the NMF decomposition class.

        :param n_components: Number of components for decomposition.
        """
        self.n_components = n_components
        self.model = NMF(n_components=n_components, init='random', random_state=42)

    def split(self, data: pd.DataFrame):
        """
        Perform NMF decomposition on the input data.

        :param data: Input DataFrame (should contain non-negative numerical values).
        :return: Factorized matrices W and H as a dictionary.
        """
        self.W = self.model.fit_transform(data)
        self.H = self.model.components_
        return {"W": self.W, "H": self.H}

    def join(self, data_parts: list):
        """
        Reconstruct the original matrix from factorized matrices.

        :param data_parts: A dictionary with "W" and "H" matrices.
        :return: Reconstructed DataFrame.
        """
        W = data_parts.get("W")
        H = data_parts.get("H")
        return pd.DataFrame(np.dot(W, H))

class PCADecomposition(BaseDecomposition):
    def __init__(self, n_components):
        """
        Initialize the PCA decomposition class.

        :param n_components: Number of principal components.
        """
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def split(self, data: pd.DataFrame):
        """
        Perform PCA decomposition on the input data.

        :param data: Input DataFrame with numerical values.
        :return: Dictionary containing the principal components and explained variance.
        """
        self.components = self.model.fit_transform(data)
        self.explained_variance = self.model.explained_variance_ratio_
        self.mean = self.model.mean_
        self.components_ = self.model.components_
        return {
            "components": self.components,
            "explained_variance": self.explained_variance,
            "mean": self.mean,
            "components_": self.components_
        }

    def join(self, data_parts: dict):
        """
        Reconstruct the original data from the principal components.

        :param data_parts: Dictionary containing the principal components and model parameters.
        :return: Reconstructed DataFrame.
        """
        components = data_parts["components"]
        components_ = data_parts["components_"]
        mean = data_parts["mean"]
        reconstructed_data = np.dot(components, components_) + mean
        return pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
from sklearn.decomposition import TruncatedSVD

class SVDDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        """
        Initialize the SVD decomposition class.

        :param n_components: Number of singular values and vectors to compute.
        """
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components)

    def split(self, data: pd.DataFrame):
        """
        Perform SVD decomposition on the input data.

        :param data: Input DataFrame with numerical values.
        :return: Dictionary containing the decomposed matrices.
        """
        self.U = self.model.fit_transform(data)
        self.Sigma = self.model.singular_values_
        self.VT = self.model.components_
        return {
            "U": self.U,
            "Sigma": self.Sigma,
            "VT": self.VT
        }

    def join(self, data_parts: dict):
        """
        Reconstruct the original data from the decomposed matrices.

        :param data_parts: Dictionary containing U, Sigma, and VT.
        :return: Reconstructed DataFrame.
        """
        U = data_parts["U"]
        Sigma = np.diag(data_parts["Sigma"])
        VT = data_parts["VT"]
        reconstructed_data = np.dot(U, np.dot(Sigma, VT))
        return pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)

from sklearn.decomposition import FastICA

class ICADecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = FastICA(n_components=n_components, random_state=42)

    def split(self, data: pd.DataFrame):
        self.S_ = self.model.fit_transform(data)
        self.A_ = self.model.mixing_
        return {"S": self.S_, "A": self.A_}

    def join(self, data_parts: dict):
        S = data_parts["S"]
        A = data_parts["A"]
        reconstructed_data = np.dot(S, A.T)
        return pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)

from sklearn.decomposition import FactorAnalysis

class FactorAnalysisDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = FactorAnalysis(n_components=n_components)

    def split(self, data: pd.DataFrame):
        self.transformed_data = self.model.fit_transform(data)
        return {"transformed_data": self.transformed_data}

    def join(self, data_parts: dict):
        # Factor Analysis does not provide a direct way to reconstruct data
        # This is a limitation of FA; you might need to handle reconstruction differently
        raise NotImplementedError("Reconstruction is not directly supported in Factor Analysis.")

from sklearn.decomposition import DictionaryLearning

class DictionaryLearningDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = DictionaryLearning(n_components=n_components, random_state=42)

    def split(self, data: pd.DataFrame):
        self.code = self.model.fit_transform(data)
        self.dictionary = self.model.components_
        return {"code": self.code, "dictionary": self.dictionary}

    def join(self, data_parts: dict):
        code = data_parts["code"]
        dictionary = data_parts["dictionary"]
        reconstructed_data = np.dot(code, dictionary)
        return pd.DataFrame(reconstructed_data, columns=data.columns)

from sklearn.decomposition import LatentDirichletAllocation

class LDADecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = LatentDirichletAllocation(n_components=n_components, random_state=42)

    def split(self, data: pd.DataFrame):
        self.doc_topic_dist = self.model.fit_transform(data)
        self.components_ = self.model.components_
        return {"doc_topic_dist": self.doc_topic_dist, "components": self.components_}

    def join(self, data_parts: dict):
        doc_topic_dist = data_parts["doc_topic_dist"]
        components = data_parts["components"]
        reconstructed_data = np.dot(doc_topic_dist, components)
        return pd.DataFrame(reconstructed_data, columns=data.columns)

def process_categorical_data(data, encoding='onehot'):
    """
    Encode categorical data into numerical format.

    :param data: Input DataFrame with categorical columns.
    :param encoding: Encoding method ('onehot' or 'label').
    :return: Encoded data as a DataFrame and the encoder object.
    """
    if encoding == 'onehot':
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(data)
        return pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(data.columns)), encoder
    elif encoding == 'label':
        label_encoders = {}
        encoded_data = pd.DataFrame()
        for column in data.columns:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        return encoded_data, label_encoders
    else:
        raise ValueError("Unsupported encoding type. Use 'onehot' or 'label'.")

def recover_categorical_data(encoded_data, encoder, encoding='onehot'):
    """
    Recover original categorical data from encoded values.

    :param encoded_data: Encoded data as a DataFrame or numpy array.
    :param encoder: Encoder object used during encoding.
    :param encoding: Encoding method used ('onehot' or 'label').
    :return: Decoded DataFrame with original categorical data.
    """
    if encoding == 'onehot':
        return pd.DataFrame(encoder.inverse_transform(encoded_data), columns=encoder.get_feature_names_out())
    elif encoding == 'label':
        recovered_data = {}
        for column, le in encoder.items():
            recovered_data[column] = le.inverse_transform(encoded_data[column].astype(int))
        return pd.DataFrame(recovered_data)
    else:
        raise ValueError("Unsupported encoding type. Use 'onehot' or 'label'.")

def test_decomposition(decomposition_class, data, **kwargs):
    print(f"\n=== Testing {decomposition_class.__name__} ===")
    print("\n=== Original Data ===")
    print(data.head())

    # Initialize and perform decomposition
    decomposer = decomposition_class(**kwargs)
    decomposed_parts = decomposer.split(data)

    # Display decomposed parts
    print("\n=== Decomposed Parts ===")
    if isinstance(decomposed_parts, dict):
        for key, value in decomposed_parts.items():
            print(f"{key} shape: {np.shape(value)}")
    else:
        print(f"Decomposed parts shape: {np.shape(decomposed_parts)}")

    # Reconstruct the data if possible
    try:
        reconstructed = decomposer.join(decomposed_parts)
        print("\n=== Reconstructed Data ===")
        print(reconstructed.head())

        # Handle any small negative values
        reconstructed[reconstructed < 0] = 0

        # Compare original and reconstructed data
        print("\n=== Comparison ===")
        original_flat = data.values.flatten()
        reconstructed_flat = reconstructed.values.flatten()
        comparison = pd.DataFrame({
            'Original': original_flat[:len(reconstructed_flat)],
            'Reconstructed': reconstructed_flat
        })
        print(comparison.head())

        # Compute reconstruction error
        mse = mean_squared_error(original_flat[:len(reconstructed_flat)], reconstructed_flat)
        print(f"\nMean Squared Error between original and reconstructed data: {mse}")
    except NotImplementedError as e:
        print(f"\n{e}")

if __name__ == "__main__":
    decomposition_classes = {
        'NMFDecomposition': NMFDecomposition,
        'PCADecomposition': PCADecomposition,
        'SVDDecomposition': SVDDecomposition,
        'ICADecomposition': ICADecomposition,
        'FactorAnalysisDecomposition': FactorAnalysisDecomposition,
        'DictionaryLearningDecomposition': DictionaryLearningDecomposition,
        #'LDADecomposition': LDADecomposition,
    }
    #test_mode = 'TruncateDecomposition'
    test_mode = 'NMFDecomposition'
    if test_mode == 'BayesianDecomposition':
        from .dataloader import DemoDataLoader
        real_data, _ = DemoDataLoader(dataset_name='covtype').load_data()
        data = real_data.sample(100)
        #### Test BayesianDecomposition
        decomposer = BayesianDecomposition()
        decomposer.split(data)
        joined_data = decomposer.join(data, 100)
        print("\nJoined Data:")
        print(joined_data)
    
    #### Test NFDecomposition
    if test_mode == 'NFDecomposition':
        data = pd.DataFrame({
            'StudentID': [1, 2, 3, 1, 2],
            'CourseID': ['CS101', 'CS102', 'CS101', 'CS102', 'CS101'],
            'StudentName': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
            'CourseTitle': ['Python', 'Java', 'Python', 'Java', 'Python'],
            'Grade': ['A', 'B', 'C', 'A', 'B']
        })
        norm = NFDecomposition(data)
        table_1nf = norm.to_1nf()
        print("Table in 1NF:")
        print(table_1nf)
        tables_2nf = norm.to_2nf()
        print("\nTables in 2NF:")
        for i, table in enumerate(tables_2nf):
            print(f"\nTable {i+1}:")
            print(table)
        tables_3nf = norm.to_3nf()
        print("\nTables in 3NF:")
        for i, table in enumerate(tables_3nf):
            print(f"\nTable {i+1}:")
            print(table)
    
    if test_mode == 'TruncateDecomposition':
        # Sample data (100 rows, 9 columns)
        data = pd.DataFrame(np.arange(900).reshape(100, 9), columns=[f"Col{i}" for i in range(1, 10)])

        # Initialize fractional truncation with 1/2 for both rows and columns
        frac_trunc_decomposer = TruncateDecomposition(0.5, 0.5)
        truncated_pieces = frac_trunc_decomposer.split(data)
        
        print("Truncated Pieces:")
        for i, piece in enumerate(truncated_pieces):
            print(f"\nPiece {i+1}:")
            print(piece)

        # Join the pieces back together
        joined_table = frac_trunc_decomposer.join(truncated_pieces)
        print("\nJoined Table:")
        print(joined_table)

    if test_mode == 'NMFDecomposition':
        # NMF requires non-negative data
        data = pd.DataFrame(np.random.rand(100, 5), columns=[f"Feature{i}" for i in range(1, 6)])
    else:
        data = pd.DataFrame(np.random.randn(100, 5), columns=[f"Feature{i}" for i in range(1, 6)])

    from sklearn.preprocessing import StandardScaler
    # Standardize data if necessary
    if test_mode in ['PCADecomposition', 'ICADecomposition', 'SVDDecomposition', 'FactorAnalysisDecomposition', 'DictionaryLearningDecomposition']:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    n_components = 2
    # Get the decomposition class
    decomposition_class = decomposition_classes.get(test_mode)
    if decomposition_class is None:
        print(f"Decomposition class for {test_mode} not found.")
    else:
        # Test the decomposition
        test_decomposition(decomposition_class, data, n_components=n_components)