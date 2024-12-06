import pandas as pd
import numpy as np
from collections import defaultdict, deque
from scipy.stats import norm
import time
import fire
import pandas as pd
import itertools
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Set

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

class OldNFDecomposition(BaseDecomposition):
    def __init__(self, data=None):
        if data:
            self.data = data
            self.tables = [data]
            self.functional_dependencies = self._find_initial_fds()
        
    def split(self, data: pd.DataFrame, nf_level: int=1):
        self.data = data.copy()
        self.tables = [data]
        self.functional_dependencies = self._find_initial_fds()
        
        if nf_level == 1:
            self.to_1nf()
        elif nf_level == 2:
            self.to_2nf()
        elif nf_level == 3:
            self.to_3nf()
        else:
            raise ValueError("Invalid normalization level. Choose 1, 2, or 3.")
        return self.tables
        
    def join(self, data_parts: list):
        """
        Joins normalized tables back together using common columns as keys.
        
        Args:
            data_parts: List of pandas DataFrames to join
            
        Returns:
            pandas.DataFrame: Joined table
        """
        if data_parts is None or len(data_parts) == 0:
            return pd.DataFrame()
        
        # Start with the first table
        result = data_parts[0].copy()
        
        # Iterate through remaining tables and join them
        for table in data_parts[1:]:
            # Find common columns between result and current table
            common_cols = list(set(result.columns) & set(table.columns))
            
            if not common_cols:
                continue  # Skip if no common columns
                
            # Perform an inner join on common columns
            result = pd.merge(
                result,
                table,
                on=common_cols,
                how='inner'
            )
            
        return result
    
    def _find_initial_fds(self):
        """Finds initial functional dependencies in the data by analyzing uniqueness."""
        columns = self.data.columns
        fds = []
        
        for col1 in tqdm(columns, desc="Finding Functional Dependencies"):
            for col2 in columns:
                if col1 != col2:
                    grouped = self.data.groupby(col1)[col2].nunique()
                    if all(grouped == 1):  # Col2 is functionally dependent on col1
                        fds.append((col1, col2))
        return fds
    
    def to_1nf(self):
        """Ensures the table is in 1NF by removing any non-atomic values."""
        for col in tqdm(self.data.columns, desc="Converting to 1NF"):
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
        
        for lhs, rhs in tqdm(minimal_cover, desc="Finding Transitive Dependencies"):
            if lhs != rhs and any(rhs != col and (col, rhs) in minimal_cover for col in self.data.columns):
                transitive_dependencies.append((lhs, rhs))
        
        tables_3nf = self._decompose_by_dependencies(transitive_dependencies)
        self.tables = self._remove_redundant_tables(tables_3nf)
        return self.tables
    
    def _find_candidate_keys(self):
        """Identify candidate keys by finding minimal sets of columns that uniquely identify rows."""
        columns = self.data.columns
        candidate_keys = []
        
        for i in tqdm(range(1, len(columns) + 1), desc="Finding Candidate Keys"):
            for combination in itertools.combinations(columns, i):
                if self.data.duplicated(subset=combination).sum() == 0:  # Unique combination
                    candidate_keys.append(combination)
        return candidate_keys
    
    def _find_minimal_cover(self):
        """Find the minimal cover of functional dependencies."""
        minimal_cover = self.functional_dependencies.copy()
        for fd in tqdm(self.functional_dependencies, desc="Finding Minimal Cover"):
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

from dataclasses import dataclass
@dataclass
class OptimizationConfig:
    # Previous parameters...
    max_key_columns: int = None
    unique_ratio_threshold: float = 0.8
    early_stop_key_size: int = 2
    min_unique_product_ratio: float = 0.9
    max_determinant_size: int = 3
    min_determinant_unique_ratio: float = 0.1
    skip_high_unique_cols: float = 0.8
    check_multi_deps: bool = True
    
    # New parameters for table filtering
    min_row_count: int = 100  # Minimum number of distinct rows in a table
    min_row_ratio: float = 0.1  # Minimum ratio of rows compared to original table
    max_similar_columns: float = 0.8  # Maximum ratio of overlapping columns between tables

original_behavior_config = OptimizationConfig(
    # Candidate key optimization parameters
    max_key_columns=None,  # No limit on key columns
    unique_ratio_threshold=0.0,  # Consider all columns as potential keys
    early_stop_key_size=999,  # Don't do early stopping
    min_unique_product_ratio=0.0,  # Don't filter based on unique product ratio

    # Functional dependency optimization parameters
    max_determinant_size=None,  # No limit on determinant size
    min_determinant_unique_ratio=0.1,  # Check all determinant combinations
    skip_high_unique_cols=1.0,  # Don't skip any columns
    check_multi_deps=True  # Check all multi-column dependencies
)
class NFDecomposition(BaseDecomposition):
    def __init__(self, optimization_config: OptimizationConfig = None):
        self.primary_key = None
        self.functional_dependencies = None
        self._column_unique_counts = None
        self.config = optimization_config or OptimizationConfig()
        # self.config = original_behavior_config # no optimization
        
    def _calculate_column_stats(self, df: pd.DataFrame):
        """Pre-calculate column statistics for optimization."""
        self._column_unique_counts = {
            col: df[col].nunique() 
            for col in df.columns
        }
        
    def _identify_candidate_keys(self, df: pd.DataFrame) -> List[Set[str]]:
        """Configurable optimized method to identify potential candidate keys."""
        if self._column_unique_counts is None:
            self._calculate_column_stats(df)
            
        columns = df.columns.tolist()
        n_rows = len(df)
        candidate_keys = []
        
        # Sort columns by unique count descending
        sorted_columns = sorted(
            columns, 
            key=lambda col: self._column_unique_counts[col],
            reverse=True
        )
        
        # First check single columns
        for col in sorted_columns:
            unique_ratio = self._column_unique_counts[col] / n_rows
            if unique_ratio >= self.config.unique_ratio_threshold:
                candidate_keys.append({col})
                
        if candidate_keys:
            return candidate_keys
            
        # Determine maximum columns to check
        max_cols = self.config.max_key_columns or len(columns)
        max_cols = min(max_cols, len(columns))
        
        # Create column combinations
        for i in tqdm(range(2, max_cols + 1), desc="Finding Candidate Keys"):
            for cols in itertools.combinations(sorted_columns[:min(len(sorted_columns), i+3)], i):
                # Quick check using unique counts product ratio
                unique_product = np.prod([self._column_unique_counts[col] for col in cols])
                if unique_product / n_rows < self.config.min_unique_product_ratio:
                    continue
                    
                if any(set(cols).issuperset(key) for key in candidate_keys):
                    continue
                
                # Check uniqueness
                combined = df[list(cols)].apply(lambda x: hash(tuple(x)), axis=1)
                if len(combined.unique()) == n_rows:
                    candidate_keys.append(set(cols))
                    
                    # Early stopping if configured
                    if i <= self.config.early_stop_key_size:
                        return candidate_keys
        
        return candidate_keys

    def _find_functional_dependencies(self, df: pd.DataFrame) -> List[Tuple[Set[str], str]]:
        """Configurable optimized method to discover functional dependencies."""
        if self._column_unique_counts is None:
            self._calculate_column_stats(df)
            
        columns = df.columns.tolist()
        n_rows = len(df)
        fds = []
        
        # Pre-calculate column groups for efficiency
        column_groups = {
            col: df.groupby(col).size().reset_index()
            for col in columns
        }
        
        pair_dependencies = defaultdict(set)
        
        # Check single-column determinants
        for det_col in tqdm(columns, desc="Finding Functional Dependencies"):
            det_unique_ratio = self._column_unique_counts[det_col] / n_rows
            
            # Skip based on unique ratio threshold
            if det_unique_ratio > self.config.skip_high_unique_cols:
                continue
                
            for dep_col in columns:
                if det_col == dep_col:
                    continue
                    
                if self._column_unique_counts[det_col] < self._column_unique_counts[dep_col]:
                    continue
                    
                grouped = df.groupby(det_col)[dep_col].nunique()
                if (grouped == 1).all():
                    fds.append((set([det_col]), dep_col))
                    pair_dependencies[det_col].add(dep_col)
        
        # Check multi-column determinants if enabled
        if self.config.check_multi_deps:
            max_det_size = min(self.config.max_determinant_size, len(columns)) if self.config.max_determinant_size else len(columns)
            
            for i in tqdm(range(2, max_det_size + 1), desc="Finding Functional Dependencies"):
                for det_cols in itertools.combinations(columns, i):
                    if all(col in pair_dependencies for col in det_cols):
                        continue
                        
                    det_unique_count = df.groupby(list(det_cols)).ngroups
                    if det_unique_count / n_rows < self.config.min_determinant_unique_ratio:
                        continue
                        
                    for dep_col in columns:
                        if dep_col in det_cols:
                            continue
                            
                        if any(pair_dependencies[det_col] for det_col in det_cols if dep_col in pair_dependencies[det_col]):
                            continue
                            
                        grouped = df.groupby(list(det_cols))[dep_col].nunique()
                        if (grouped == 1).all():
                            fds.append((set(det_cols), dep_col))
        
        return fds

    def _is_atomic(self, df: pd.DataFrame) -> bool:
        """Check if all values in the DataFrame are atomic."""
        for column in df.columns:
            # Check for lists, dicts, or other complex types
            if df[column].apply(lambda x: isinstance(x, (list, dict, set))).any():
                return False
        return True

    def _to_1nf(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Convert to First Normal Form."""
        if self._is_atomic(df):
            return [df]
        
        # Handle non-atomic values by creating separate tables
        result_tables = []
        main_table = df.copy()
        
        for column in df.columns:
            if df[column].apply(lambda x: isinstance(x, (list, dict, set))).any():
                # Create a new table for this column
                new_table = pd.DataFrame(columns=['id', column])
                rows = []
                
                for idx, value in df[column].items():
                    if isinstance(value, (list, set)):
                        for item in value:
                            rows.append({'id': idx, column: item})
                    elif isinstance(value, dict):
                        rows.append({'id': idx, **value})
                    else:
                        rows.append({'id': idx, column: value})
                
                new_table = pd.DataFrame(rows)
                result_tables.append(new_table)
                main_table = main_table.drop(columns=[column])
        
        result_tables.insert(0, main_table)
        return result_tables

    def _to_2nf(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Convert to Second Normal Form."""
        # First, ensure 1NF
        tables_1nf = self._to_1nf(df)
        if not tables_1nf:
            return []

        main_table = tables_1nf[0]
        result_tables = []

        # Identify candidate keys and functional dependencies
        candidate_keys = self._identify_candidate_keys(main_table)
        if not candidate_keys:
            return tables_1nf

        self.primary_key = min(candidate_keys, key=len)
        self.functional_dependencies = self._find_functional_dependencies(main_table)

        # Identify partial dependencies
        partial_deps = {}
        for determinant, dependent in tqdm(self.functional_dependencies, desc="Finding Partial Dependencies"):
            if any(determinant.issubset(key) and not determinant.issuperset(key) 
                  for key in candidate_keys):
                if frozenset(determinant) not in partial_deps:
                    partial_deps[frozenset(determinant)] = set()
                partial_deps[frozenset(determinant)].add(dependent)

        # Create new tables for partial dependencies
        main_columns = set(main_table.columns)
        for determinant, dependents in tqdm(partial_deps.items(), desc="Creating Partial Tables"):
            if dependents:
                new_table_columns = list(determinant) + list(dependents)
                new_table = main_table[new_table_columns].drop_duplicates()
                result_tables.append(new_table)
                main_columns -= dependents

        # Keep remaining columns in main table
        main_table = main_table[list(main_columns)]
        result_tables.insert(0, main_table)
        
        return result_tables + tables_1nf[1:]

    def _is_table_significant(self, table: pd.DataFrame, original_row_count: int) -> bool:
        """Check if a table meets the significance criteria."""
        distinct_rows = len(table.drop_duplicates())
        row_ratio = distinct_rows / original_row_count
        
        return (distinct_rows >= self.config.min_row_count and 
                row_ratio >= self.config.min_row_ratio)

    def _filter_similar_tables(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Filter out tables that are too similar to others."""
        if not tables:
            return []
            
        filtered_tables = [tables[0]]
        filtered_indices = {0}  # Keep track of indices we want to keep
        
        for i, current_table in enumerate(tables[1:], 1):
            should_add = True
            to_remove = []
            
            for j, existing_table in enumerate(filtered_tables):
                if self._are_tables_similar(current_table, existing_table):
                    # If similar, keep the one with more distinct rows
                    current_rows = len(current_table.drop_duplicates())
                    existing_rows = len(existing_table.drop_duplicates())
                    
                    if current_rows > existing_rows:
                        # Mark this index for removal
                        to_remove.append(j)
                        should_add = True
                    else:
                        should_add = False
                    break
                    
            # Remove marked tables
            for idx in reversed(to_remove):  # Remove from end to avoid index shifting
                filtered_tables.pop(idx)
                
            if should_add:
                filtered_tables.append(current_table)
                filtered_indices.add(i)
                
        return filtered_tables

    def _are_tables_similar(self, table1: pd.DataFrame, table2: pd.DataFrame) -> bool:
        """Check if two tables have too many overlapping columns."""
        cols1 = set(table1.columns)
        cols2 = set(table2.columns)
        overlap = len(cols1.intersection(cols2))
        min_cols = min(len(cols1), len(cols2))
        
        if min_cols == 0:
            return False
            
        # Check column overlap ratio
        overlap_ratio = overlap / min_cols
        
        # Also check if tables contain mostly the same data
        if overlap_ratio >= self.config.max_similar_columns:
            # Compare the overlapping columns data
            common_cols = list(cols1.intersection(cols2))
            if common_cols:
                t1_data = table1[common_cols].values
                t2_data = table2[common_cols].values
                return np.array_equal(np.sort(t1_data, axis=0), np.sort(t2_data, axis=0))
                
        return False

    def _to_3nf(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Convert to Third Normal Form with optimized table generation and filtering."""
        # First, ensure 2NF
        tables_2nf = self._to_2nf(df)
        if not tables_2nf:
            return []

        main_table = tables_2nf[0]
        result_tables = []
        original_row_count = len(df)

        # Find transitive dependencies
        transitive_deps = defaultdict(set)
        processed_deps = set()
        if not self.functional_dependencies:
            return tables_2nf

        # Group related dependencies
        for det1, dep1 in self.functional_dependencies:
            for det2, dep2 in self.functional_dependencies:
                if dep1 in det2 and dep2 not in det1:
                    key = frozenset(det1)
                    transitive_deps[key].add(dep1)
                    transitive_deps[key].add(dep2)
                    processed_deps.add(dep1)
                    processed_deps.add(dep2)

        # Create consolidated tables for transitive dependencies
        main_columns = set(main_table.columns)
        for determinant, dependents in transitive_deps.items():
            if dependents:
                new_table_columns = list(determinant) + list(dependents)
                new_table = main_table[new_table_columns].drop_duplicates()
                
                if self._is_table_significant(new_table, original_row_count):
                    result_tables.append(new_table)
                    main_columns -= dependents

        # Handle remaining functional dependencies
        remaining_deps = defaultdict(set)
        for det, dep in self.functional_dependencies:
            if dep not in processed_deps:
                key = frozenset(det)
                remaining_deps[key].add(dep)
                processed_deps.add(dep)

        # Create tables for remaining dependencies
        for determinant, dependents in remaining_deps.items():
            if dependents:
                new_table_columns = list(determinant) + list(dependents)
                new_table = main_table[new_table_columns].drop_duplicates()
                
                if self._is_table_significant(new_table, original_row_count):
                    result_tables.append(new_table)
                    main_columns -= dependents

        # Keep remaining columns in main table
        if main_columns:
            main_table = main_table[list(main_columns)]
            if self._is_table_significant(main_table, original_row_count):
                result_tables.insert(0, main_table)

        # Remove duplicate and similar tables
        result_tables = self._filter_similar_tables(result_tables)

        return result_tables + tables_2nf[1:]
    
    # def _to_3nf(self, df: pd.DataFrame) -> List[pd.DataFrame]:
    #     """Convert to Third Normal Form."""
    #     # First, ensure 2NF
    #     tables_2nf = self._to_2nf(df)
    #     if not tables_2nf:
    #         return []

    #     main_table = tables_2nf[0]
    #     result_tables = []

    #     # Find transitive dependencies
    #     transitive_deps = {}
    #     for det1, dep1 in tqdm(self.functional_dependencies, desc="Finding Transitive Dependencies"):
    #         for det2, dep2 in self.functional_dependencies:
    #             if dep1 in det2 and dep2 not in det1:
    #                 if frozenset(det1) not in transitive_deps:
    #                     transitive_deps[frozenset(det1)] = set()
    #                 transitive_deps[frozenset(det1)].add(dep2)

    #     # Create new tables for transitive dependencies
    #     main_columns = set(main_table.columns)
    #     for determinant, dependents in tqdm(transitive_deps.items(), desc="Creating Transitive Tables"):
    #         if dependents:
    #             new_table_columns = list(determinant) + list(dependents)
    #             new_table = main_table[new_table_columns].drop_duplicates()
    #             result_tables.append(new_table)
    #             main_columns -= dependents

    #     # Keep remaining columns in main table
    #     main_table = main_table[list(main_columns)]
    #     result_tables.insert(0, main_table)
        
    #     return result_tables + tables_2nf[1:]

    def split(self, data: pd.DataFrame, nf_level: int = 1) -> List[pd.DataFrame]:
        self.original_data_columns = data.columns.tolist()
        """Returns the nf-level normalized decomposed tables."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if nf_level not in [1, 2, 3]:
            raise ValueError("NF level must be 1, 2, or 3")
        
        if nf_level == 1:
            return self._to_1nf(data)
        elif nf_level == 2:
            return self._to_2nf(data)
        else:  # nf_level == 3
            return self._to_3nf(data)


    def join(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """Joins the normalized tables back together."""
        if not tables:
            return pd.DataFrame()
        
        result = tables[0]
        for table in tqdm(tables[1:], desc="Joining Tables"):
            common_cols = list(set(result.columns) & set(table.columns))
            if common_cols:
                result = result.merge(table, on=common_cols, how='left')
        # remaining_tables = tables[1:]
        # tqdm_remaining_tables = tqdm(remaining_tables, desc="Joining Tables")
        # while remaining_tables:
        #     # table_with_most_common_cols = max(remaining_tables, key=lambda x: len(set(result.columns) & set(x.columns)))
        #     table = min(remaining_tables, key=lambda x: len(set(result.columns)))
        #     common_cols = list(set(result.columns) & set(table.columns))
        #     if common_cols:
        #         result = result.merge(table, on=common_cols, how='left')
        #         remaining_tables.remove(table)
        #     else:
        #         remaining_tables.remove(table)
        #     tqdm_remaining_tables.update(1)
                
        # change column order to match original data
        result = result[self.original_data_columns]
            
        return result
   
class TruncateDecomposition(BaseDecomposition):
    def __init__(self, row_fraction: float=None, col_fraction: float=None):
        """
        Initialize with row and column fractions.
        :param row_fraction: Fraction for row partitioning (e.g., 0.5 for half, resulting in 2 parts).
        :param col_fraction: Fraction for column partitioning (e.g., 0.5 for half, resulting in 2 parts).
        """
        self.row_fraction = row_fraction
        self.col_fraction = col_fraction

    def split(self, data: pd.DataFrame, row_fraction: float=None, col_fraction: float=None):
        """
        Split the DataFrame into four partitions:
        - First half of rows with first half of columns
        - First half of rows with second half of columns
        - Second half of rows with first half of columns
        - Second half of rows with second half of columns
        """
        row_fraction = row_fraction or self.row_fraction
        col_fraction = col_fraction or self.col_fraction
        assert row_fraction is not None and col_fraction is not None, "Row and column fractions must be provided."
        num_rows, num_columns = data.shape
        row_midpoint = int(num_rows * row_fraction)
        col_midpoint = int(num_columns * col_fraction)

        # Create 4 parts based on row and column midpoints
        part1 = data.iloc[:row_midpoint, :col_midpoint]    # Top-left
        part2 = data.iloc[:row_midpoint, col_midpoint:]    # Top-right
        part3 = data.iloc[row_midpoint:, :col_midpoint]    # Bottom-left
        part4 = data.iloc[row_midpoint:, col_midpoint:]    # Bottom-right

        return [part1, part2, part3, part4]

    def num_rows_to_sample_per_subtable(self, num_rows: int):
        """
        Calculate the number of rows to sample per subtable based on the row fraction.
        """
        row_midpoint = int(num_rows * self.row_fraction)
        return [row_midpoint, row_midpoint, num_rows - row_midpoint, num_rows - row_midpoint]
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
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}
        
        self.W = self.model.fit_transform(data)
        self.H = self.model.components_
        # convert to pandas DataFrame
        decomposed_table = pd.DataFrame(self.W, columns=[f"Component{i+1}" for i in range(self.n_components)])
        return [decomposed_table]

    def join(self, tables: pd.DataFrame):
        """
        Reconstruct the original matrix from factorized matrices.

        :param data_parts: A dictionary with "W" and "H" matrices.
        :return: Reconstructed DataFrame.
        """
        # W = data_parts.get("W")
        # H = data_parts.get("H")
        W = tables
        if isinstance(W, list):
            W = W[0]
        H = self.H
        # convert to numpy array
        if isinstance(W, pd.DataFrame):
            W = W.to_numpy()
        if isinstance(H, pd.DataFrame):
            H = H.to_numpy()
            
        reconstructed_data = self.model.inverse_transform(W)
            
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df

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
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}

        self.components = self.model.fit_transform(data)
        self.explained_variance = self.model.explained_variance_ratio_
        self.mean = self.model.mean_
        self.components_ = self.model.components_
        # convert to pandas DataFrame
        decomposed_table = pd.DataFrame(self.components, columns=[f"PrincipalComponent{i+1}" for i in range(self.n_components)])
        return [decomposed_table]
        # return {
        #     "components": self.components,
        #     "explained_variance": self.explained_variance,
        #     "mean": self.mean,
        #     "components_": self.components_
        # }

    def join(self, tables: pd.DataFrame):
        """
        Reconstruct the original data from the principal components.

        :param components: DataFrame containing the principal components.
        :return: Reconstructed DataFrame.
        """
        components = tables
        if isinstance(components, list):
            components = components[0]
        # convert to numpy array
        if isinstance(components, pd.DataFrame):
            components = components.to_numpy()
        reconstructed_data = np.dot(components, self.model.components_) + self.model.mean_
        
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df
    
    def join_data_parts(self, data_parts: dict):
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
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}
        
        self.U = self.model.fit_transform(data)
        self.Sigma = self.model.singular_values_
        self.VT = self.model.components_
        # convert to pandas DataFrame
        U_table = pd.DataFrame(self.U, columns=[f"U{i+1}" for i in range(self.n_components)])
        return [U_table]
        # return {
        #     "U": self.U,
        #     "Sigma": self.Sigma,
        #     "VT": self.VT
        # }

    def join(self, tables: pd.DataFrame):
        if isinstance(tables, list):
            U = tables[0]
        else:
            U = tables
        if isinstance(U, pd.DataFrame):
            U = U.to_numpy()
        # reconstructed_data = self.model.inverse_transform(U)
        # Dongfu: U has already been applied to Sigma, no need to multiply again
        reconstructed_data = U @ self.VT
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df
    
    
    def join_data_parts(self, data_parts: dict):
        """
        Reconstruct the original data from the decomposed matrices.

        :param data_parts: Dictionary containing U, Sigma, and VT.
        :return: Reconstructed DataFrame.
        """
        U = data_parts["U"]
        Sigma = np.diag(data_parts["Sigma"])
        VT = data_parts["VT"]
        # Dongfu: U has already been applied to Sigma, no need to multiply again
        reconstructed_data = np.dot(U, VT)
        return pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)

from sklearn.decomposition import FastICA

class ICADecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = FastICA(n_components=n_components, random_state=42)

    def split(self, data: pd.DataFrame):
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}
        
        self.S_ = self.model.fit_transform(data)
        self.A_ = self.model.mixing_
        # convert to pandas DataFrame
        S_table = pd.DataFrame(self.S_, columns=[f"S{i+1}" for i in range(self.n_components)])
        return [S_table]

    def join(self, tables: pd.DataFrame):
        if isinstance(tables, list):
            S = tables[0]
        else:
            S = tables
        if isinstance(S, pd.DataFrame):
            S = S.to_numpy()
        reconstructed_data = self.model.inverse_transform(S)
        
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df
    
    def join_data_parts(self, data_parts: dict):
        S = data_parts["S"]
        A = data_parts["A"]
        # reconstructed_data = np.dot(S, A.T) # not correct
        reconstructed_data = self.model.inverse_transform(S)
        return pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)

from sklearn.decomposition import FactorAnalysis

class FactorAnalysisDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = FactorAnalysis(n_components=n_components)

    def split(self, data: pd.DataFrame):
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}
        
        self.X_transformed = self.model.fit_transform(data)
        table = pd.DataFrame(self.X_transformed, columns=[f"Factor{i+1}" for i in range(self.n_components)])
        return [table]

    def join(self, tables: pd.DataFrame):
        if isinstance(tables, list):
            X_transformed = tables[0]
        else:
            X_transformed = tables
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed = X_transformed.to_numpy()
        # reconstructed_data = self.model.inverse_transform(X_transformed)
        reconstructed_data = np.dot(X_transformed, self.model.components_) + self.model.mean_
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df

from sklearn.decomposition import DictionaryLearning

class DictionaryLearningDecomposition(BaseDecomposition):
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = DictionaryLearning(n_components=n_components, random_state=42)

    def split(self, data: pd.DataFrame):
        # for the string columns, we need to convert them to numerical values, and keep a index to convert back
        self.category_mappings = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Create categorical and store the mapping
                categorical = pd.Categorical(data[col])
                data[col] = categorical.codes
                self.category_mappings[col] = dict(enumerate(categorical.categories))
        # self.column_data_types = data.dtypes
        self.column_data_types = {col: str(data[col].dtype) for col in data.columns}
        
        self.code = self.model.fit_transform(data)
        self.dictionary = self.model.components_
        # convert to pandas DataFrame
        code_table = pd.DataFrame(self.code, columns=[f"Code{i+1}" for i in range(self.n_components)])
        return [code_table]

    def join(self, tables: pd.DataFrame):
        if isinstance(tables, list):
            code = tables[0]
        else:
            code = tables
        if isinstance(code, pd.DataFrame):
            code = code.to_numpy()
        reconstructed_data = np.dot(code, self.model.components_)
        
        joined_df = pd.DataFrame(reconstructed_data, columns=self.model.feature_names_in_)
        for col in self.column_data_types:
            if "int" in self.column_data_types[col]:
                joined_df[col] = joined_df[col].round(0)
        joined_df = joined_df.astype(self.column_data_types)
        # convert back to original data types for all columns, using self.column_data_types, not the fucking self.category_mappings, 
        for col in self.category_mappings:
            joined_df[col] = joined_df[col].map(self.category_mappings[col])
        return joined_df

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
        
        
def main(
    dataset_name: str = 'adult',
    decomposition_method: str = 'PCADecomposition',
    sample_size: int = 1000,
    decomposition_init_kwargs: dict = {},
    decomposition_kwargs: dict = {}
):
    from collections import defaultdict
    decomposition_classes = {
        'BayesianDecomposition': BayesianDecomposition,
        'NFDecomposition': NFDecomposition,
        "TruncateDecomposition": TruncateDecomposition,
        
        'NMFDecomposition': NMFDecomposition,
        'PCADecomposition': PCADecomposition,
        'SVDDecomposition': SVDDecomposition,
        'ICADecomposition': ICADecomposition,
        'FactorAnalysisDecomposition': FactorAnalysisDecomposition,
        'DictionaryLearningDecomposition': DictionaryLearningDecomposition,
    }
    default_decomposition_kwargs = defaultdict(lambda: {})
    default_decomposition_kwargs.update({
        "NFDecomposition": {"nf_level": 3},
        "TruncateDecomposition": {"row_fraction": 0.5, "col_fraction": 0.5},
    })
    default_decomposition_init_kwargs = defaultdict(lambda: {})
    default_decomposition_init_kwargs.update({
        "PCADecomposition": {"n_components": 8},
        "ICADecomposition": {"n_components": 8},
        "NMFDecomposition": {"n_components": 8},
        "SVDDecomposition": {"n_components": 8},
        "FactorAnalysisDecomposition": {"n_components": 8},
        "DictionaryLearningDecomposition": {"n_components": 8},
    })
        
    if not decomposition_kwargs:
        decomposition_kwargs = default_decomposition_kwargs.get(decomposition_method, {})
    if not decomposition_init_kwargs:
        decomposition_init_kwargs = default_decomposition_init_kwargs.get(decomposition_method, {})
        
    from dataloader import DemoDataLoader
    real_data, _ = DemoDataLoader(dataset_name=dataset_name).load_data()
    data = real_data.sample(sample_size, random_state=42)
    print("Example Data:")
    print(data)
    
    assert decomposition_method in decomposition_classes, f"Invalid test mode: {decomposition_method}"
    decomposition_class = decomposition_classes[decomposition_method]
    decomposer = decomposition_class(**decomposition_init_kwargs)
    
    
    # decompose
    start_time = time.time()
    decomposed_tables = decomposer.split(data, **decomposition_kwargs)
    end_time = time.time()
    print(f"\nDecomposition Time: {end_time - start_time:.4f} seconds")
    print(f"\nDecomposed Tables ({decomposition_method}):")
    for i, table in enumerate(decomposed_tables):
        print(f"\nTable {i+1}:")
        print(table)
    
    # join
    start_time = time.time()
    joined_data = decomposer.join(decomposed_tables)
    end_time = time.time()
    print(f"\nJoin Time: {end_time - start_time:.4f} seconds")
    print("\nJoined Data:")
    print(joined_data)
    
    
    # if test_mode == 'BayesianDecomposition':
    #     #### Test BayesianDecomposition
    #     decomposer = BayesianDecomposition()
    #     decomposer.split(data)
    #     joined_data = decomposer.join(data, 100)
    #     print("\nJoined Data:")
    #     print(joined_data)
    
    # #### Test NFDecomposition
    # if test_mode == 'NFDecomposition':
    #     data = pd.DataFrame({
    #         'StudentID': [1, 2, 3, 1, 2],
    #         'CourseID': ['CS101', 'CS102', 'CS101', 'CS102', 'CS101'],
    #         'StudentName': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
    #         'CourseTitle': ['Python', 'Java', 'Python', 'Java', 'Python'],
    #         'Grade': ['A', 'B', 'C', 'A', 'B']
    #     })
    #     norm = NFDecomposition(data)
    #     table_1nf = norm.to_1nf()
    #     print("Table in 1NF:")
    #     print(table_1nf)
    #     tables_2nf = norm.to_2nf()
    #     print("\nTables in 2NF:")
    #     for i, table in enumerate(tables_2nf):
    #         print(f"\nTable {i+1}:")
    #         print(table)
    #     tables_3nf = norm.to_3nf()
    #     print("\nTables in 3NF:")
    #     for i, table in enumerate(tables_3nf):
    #         print(f"\nTable {i+1}:")
    #         print(table)
    
    # if test_mode == 'TruncateDecomposition':
    #     # Sample data (100 rows, 9 columns)
    #     data = pd.DataFrame(np.arange(900).reshape(100, 9), columns=[f"Col{i}" for i in range(1, 10)])

    #     # Initialize fractional truncation with 1/2 for both rows and columns
    #     frac_trunc_decomposer = TruncateDecomposition(0.5, 0.5)
    #     truncated_pieces = frac_trunc_decomposer.split(data)
        
    #     print("Truncated Pieces:")
    #     for i, piece in enumerate(truncated_pieces):
    #         print(f"\nPiece {i+1}:")
    #         print(piece)

    #     # Join the pieces back together
    #     joined_table = frac_trunc_decomposer.join(truncated_pieces)
    #     print("\nJoined Table:")
    #     print(joined_table)

    # if test_mode == 'NMFDecomposition':
    #     # NMF requires non-negative data
    #     data = pd.DataFrame(np.random.rand(100, 5), columns=[f"Feature{i}" for i in range(1, 6)])
    # else:
    #     data = pd.DataFrame(np.random.randn(100, 5), columns=[f"Feature{i}" for i in range(1, 6)])

    # from sklearn.preprocessing import StandardScaler
    # # Standardize data if necessary
    # if test_mode in ['PCADecomposition', 'ICADecomposition', 'SVDDecomposition', 'FactorAnalysisDecomposition', 'DictionaryLearningDecomposition']:
    #     scaler = StandardScaler()
    #     data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # n_components = 2
    # # Get the decomposition class
    # decomposition_class = decomposition_classes.get(test_mode)
    # if decomposition_class is None:
    #     print(f"Decomposition class for {test_mode} not found.")
    # else:
    #     # Test the decomposition
    #     test_decomposition(decomposition_class, data, n_components=n_components)
        
        
if __name__ == "__main__":
    fire.Fire(main)