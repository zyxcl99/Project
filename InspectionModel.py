import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns

class DataInspection:
    def __init__(self):
        self.df = None  

    def load_csv(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded from {file_path}.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    def summarize_columns_variables(self):
        print(f"{'Variable':<15}{'Type':<10}{'Mean / Median / Mode':<20}{'Kurtosis':<10}{'Skewness':<10}")
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                var_type = 'Nominal'
            elif (self.df[col] % 1).eq(0).all() and self.df[col].dtype != 'object':
                var_type = 'Ordinal'
            else:
                var_type = 'Ratio'
            if var_type == 'Nominal':
                measure = str(self.df[col].mode()[0])
                kurt = 'NA'
                skewness = 'NA'
            elif var_type == 'Ordinal':
                measure = f"{self.df[col].median():.2f}"
                kurt = f"{kurtosis(self.df[col].dropna()):.2f}"
                skewness = f"{skew(self.df[col].dropna()):.2f}"
            else: 
                measure = f"{self.df[col].mean():.2f}"
                kurt = f"{kurtosis(self.df[col].dropna()):.2f}"
                skewness = f"{skew(self.df[col].dropna()):.2f}"
            print(f"{col:<15}{var_type:<10}{measure:<20}{kurt:<10}{skewness:<10}")
    def lis_variables_available_distribution(self):
        numerical_vars = self.df.select_dtypes(include=['number']).columns.tolist()
        print("Numerical Variables available for distribution analysis:")
        for var in numerical_vars:
            print(var)

    def show_distribution(self, variable):
        if variable in self.df.columns:
            try:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.df[variable], kde=True)
                plt.title(f"Distribution of {variable}")
                plt.xlabel(variable)
                plt.ylabel("Frequency")
                plt.show()
            except Exception as e:
                print(f"Error showing distribution: {e}")
        else:
            print("Variable not found in the dataset.")


    def plot_histogram(self):
        numeric_cols = self.numeric_columns()
        print(f"Available numeric columns: {numeric_cols}")
        col = input("Choose column for histogram:")
        self.df[col].hist()
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self):
        numeric_cols = self.numeric_columns()
        print(f"Available numeric columns: {numeric_cols}")
        col = input("Choose column for boxplot:")
        self.df.boxplot(column=col)
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Values')
        plt.show()
    def plot_scatter(self):
        numeric_cols = self.numeric_columns()
        print(f"Available numeric columns: {numeric_cols}")
        x_col = input("Choose x-column for scatter plot: ")
        y_col = input("Choose y-column for scatter plot: ")
        self.df.plot.scatter(x=x_col, y=y_col)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    def plot_bar_chart(self,):
        cols = self.df.columns
        print(f"Available columns: {cols}")
        col = input("Choose column for bar-chart:")
        self.df[col].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    
    def numeric_columns(self):
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]