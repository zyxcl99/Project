import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import scipy.stats as ss
class DataAnalysis:
    def __init__(self):
        self.df = pd.DataFrame()

    def dataset_loading(self, file_path):
        self.df = pd.read_csv(file_path)

    def list_suitable_variables(self, test_type):
        print(f"Available variables for {test_type}:")
        if test_type == 'ANOVA' or test_type == 't-Test':
            print("Continuous (interval/ratio) variables:")
            for var in self.df.select_dtypes(include=['int64', 'float64']).columns:
                print(var)
            print("Categorical (ordinal/nominal) variables:")
            for var in self.df.select_dtypes(include=['object']).columns:
                
                if self.df[var].dtype == 'object' or self.df[var].nunique() < 10 or (self.df[var] % 1).eq(0).all() and self.df[var].dtype != 'object':
                    print(var)
        elif test_type == 'Regression':
            print("Continuous (interval/ratio) variables:")
            for var in self.df.select_dtypes(include=['int64', 'float64']).columns:
                print(var)
        elif test_type == 'Chi-Square':
            print("Categorical (nominal/ordinal) variables:")
            for var in self.df.select_dtypes(include=['object']).columns:
                if self.df[var].dtype == 'object' or self.df[var].nunique() < 10 or (self.df[var] % 1).eq(0).all() and self.df[var].dtype != 'object':
                    print(var)
        elif test_type == 'Sentiment':
            print("Text variables:")
            for var in self.df.select_dtypes(include=['object']).columns:
                print(var)
    def anova_test(self, continuous_var, categorical_var):
        print(f"Performing ANOVA for '{continuous_var}' and '{categorical_var}'...")
        try:
            normal_dist = self.check_normality(continuous_var)
            self.plot_qq_histogram(continuous_var)
            if not normal_dist:
                print(f"'{continuous_var}' is not normally distributed, performing Kruskal-Wallis Test...")
                stat, p_value = ss.kruskal(*(self.df[self.df[categorical_var] == level][continuous_var]
                                         for level in self.df[categorical_var].unique()))
                print(f"Kruskal-Wallis Result:\nStatistic: {stat}\np-value: {p_value}")
                if p_value < 0.05:
                    print("Null Hypothesis Rejected: Statistically significant difference.")
                else:
                    print("Failed to Reject Null Hypothesis.")
            else:
                f_value, p_value = ss.f_oneway(*(self.df[self.df[categorical_var] == level][continuous_var]
                                                 for level in self.df[categorical_var].unique()))
                print(f"ANOVA Result:\nF-value: {f_value}\np-value: {p_value}")
                if p_value < 0.05:
                    print("Null Hypothesis Rejected: There is a significant difference.")
                else:
                    print("Failed to Reject Null Hypothesis: No significant difference.")
        except Exception as e:
            print("Error performing ANOVA test:", e)
    def plot_anova_boxplot(self, data, continuous_var, categorical_var, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_var, y=continuous_var, data=data)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()  
        plt.show()

    def t_test_mannwhitney(self, group_var, response_var):
        print("Conduct t-Test / Mann-Whitney U test")
        self.plot_qq_histogram(group_var)
        self.plot_qq_histogram(response_var)
        if self.check_normality(response_var):
            print("Conducting t-Test...")
            t_stat, p_value = stats.ttest_ind(self.df[self.df[group_var] == self.df[group_var].unique()[0]][response_var],
                                              self.df[self.df[group_var] == self.df[group_var].unique()[1]][response_var])
            print(' ')
            print(f"T-statistic: {t_stat}, P-value: {p_value}")
            if p_value < 0.05:
                print("Null Hypothesis Rejected: There is a significant difference.")
            else:
                print("Failed to Reject Null Hypothesis: No significant difference.")
        else:
            print("Not fit t-Test Conducting Mann-Whitney U test...")
            u_stat, p_value = stats.mannwhitneyu(self.df[self.df[group_var] == self.df[group_var].unique()[0]][response_var],
                                                  self.df[self.df[group_var] == self.df[group_var].unique()[1]][response_var])
            print(f"U-statistic: {u_stat}, P-value: {p_value}")
            if p_value < 0.05:
                print("Null Hypothesis Rejected: There is a significant difference.")
            else:
                print("Failed to Reject Null Hypothesis: No significant difference.")

    def chi_square_test(self, var1, var2):
        print("Chi-square Test")
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f'''Chi-square statistic: {chi2}, 
P-value: {p}
dof: {dof}
expected: {expected}''')
        if p < 0.05:
            print("Null Hypothesis Rejected: There is a significant association.")
        else:
            print("Failed to Reject Null Hypothesis: No significant association.")

    def regression_analysis(self, x_var, y_var):
        """
        Perform linear regression between two interval variables.
        """
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()
        self.plot_regression(X, Y)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"P-value: {p_value:.15f}")
        print(f"Standard error: {std_err:.4f}")
    def textblob_sentiment_analysis(self, data):
        scores = data.apply(lambda remark: TextBlob(remark).sentiment.polarity)
        sentiments = data.apply(lambda remark: 'Positive' if TextBlob(remark).sentiment.polarity >= 0 else 'Negative')
        subjectivity = data.apply(lambda remark: TextBlob(remark).sentiment.subjectivity)
        print(scores, sentiments,subjectivity)
        return scores, sentiments, subjectivity
    
    def check_normality(self, variable):
        stat, p_value = stats.shapiro(self.df[variable].dropna())
        print(f"Shapiro-Wilk Test: Statistic={stat}, P-value={p_value}")
        return p_value > 0.05
    
    def plot_qq_histogram(self, title):
        """This function simply plots Q-Q and histogram for the chosen variable."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sm.qqplot(self.df[title], line='s', ax=plt.gca())
        plt.title(f'Q-Q Plot of {title}')
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.df[title], kde=True)
        plt.title(f'Histogram of {title}')
        
        plt.show()
    def plot_regression(self, X, Y):
        try:
            if isinstance(X, pd.Series):
                X = X.values.reshape(-1, 1)
            elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
                X = X.values
            model = LinearRegression()
            model.fit(X, Y)
            Y_pred = model.predict(X)
            plt.figure(figsize=(10, 6))
            plt.scatter(X, Y, color='blue', label='Actual data')
            plt.plot(X, Y_pred, color='red', linewidth=2, label='Fitted line')
            plt.legend()
            plt.title('Linear Regression Plot')
            plt.xlabel('Independent Variable(s)')
            plt.ylabel('Dependent Variable')
            plt.show()
            print(f"Coefficients: {model.coef_.flatten()}")
            print(f"Intercept: {model.intercept_}")
            r_squared = model.score(X, Y)
            print(f"R-squared: {r_squared}")
        except Exception as e:
            print("Error plotting regression:", e)
    def plot_heatmap(self, category1, category2, title):
        cross_tab = pd.crosstab(self.df[category1], self.df[category2])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g', cbar_kws={'label': 'Count'})
        plt.title(title)
        plt.xlabel(category2)
        plt.ylabel(category1)
        plt.show()