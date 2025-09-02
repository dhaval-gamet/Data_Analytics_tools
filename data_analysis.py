import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import missingno as msno
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sweetviz as sv
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class DataAnalysisTool:
    def __init__(self):
        self.df = None
        self.analysis_result = None
    
    def load_data(self, file, filename):
        """Load data from file object"""
        file_extension = filename.split('.')[-1].lower()
        try:
            if file_extension == 'csv':
                self.df = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                self.df = pd.read_excel(file)
            elif file_extension == 'json':
                self.df = pd.read_json(file)
            else:
                return False, "Unsupported file format! Please upload CSV, Excel, or JSON files."
            
            return True, f"File '{filename}' successfully loaded! Data shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns"
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def auto_data_cleaning_suggestions(self):
        """Automated data cleaning suggestions"""
        if self.df is None:
            return "No data loaded!"
        
        suggestions = []
        suggestions.append("="*60)
        suggestions.append("Automated Data Cleaning Suggestions")
        suggestions.append("="*60)
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        if missing.sum() > 0:
            suggestions.append("Missing values found:")
            for col in missing[missing > 0].index:
                suggestions.append(f" - {col}: {missing[col]} missing values ({missing_percent[col]:.2f}%)")
                # Suggestion based on missing percentage
                if missing_percent[col] > 50:
                    suggestions.append(f"  Suggestion: Consider removing '{col}' column (more than 50% missing values)")
                elif missing_percent[col] > 10:
                    suggestions.append(f"  Suggestion: Use advanced imputation techniques for '{col}' column")
                else:
                    suggestions.append(f"  Suggestion: Fill missing values in '{col}' column with mean/median/mode")
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            suggestions.append(f"Duplicate rows: {duplicates}")
            suggestions.append("Suggestion: Consider removing duplicate rows")
        
        # Data type suggestions
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if it might be a date column
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    suggestions.append(f"Suggestion: Convert '{col}' column to datetime type")
                except:
                    unique_ratio = self.df[col].nunique() / len(self.df)
                    if unique_ratio < 0.05:
                        suggestions.append(f"Suggestion: Convert '{col}' column to categorical type")
        
        # Outlier detection for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                suggestions.append(f"Suggestion: Found {outlier_count} outliers in '{col}' column. Consider: log transformation, capping, or removal")
        
        return "\n".join(suggestions)
    
    def clean_data(self, strategies=None):
        """Automated data cleaning with customizable strategies"""
        if self.df is None:
            return False, "No data loaded!"
        
        df_clean = self.df.copy()
        messages = []
        
        # Default strategies if none provided
        if strategies is None:
            strategies = {
                'remove_duplicates': True,
                'handle_missing_numeric': 'median',
                'handle_missing_categorical': 'mode',
                'remove_high_missing': True,
                'threshold_high_missing': 50,
                'convert_to_categorical': True,
                'categorical_threshold': 0.05
            }
        
        # Remove duplicates
        if strategies['remove_duplicates']:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = initial_rows - len(df_clean)
            if removed > 0:
                messages.append(f"{removed} duplicate rows removed")
        
        # Remove columns with high percentage of missing values
        if strategies['remove_high_missing']:
            missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
            cols_to_remove = missing_percent[missing_percent > strategies['threshold_high_missing']].index
            df_clean = df_clean.drop(columns=cols_to_remove)
            if len(cols_to_remove) > 0:
                messages.append(f"{len(cols_to_remove)} columns removed (high missing value percentage): {list(cols_to_remove)}")
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategies['handle_missing_numeric'] == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategies['handle_missing_numeric'] == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategies['handle_missing_numeric'] == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif strategies['handle_missing_numeric'] == 'remove':
                    df_clean = df_clean.dropna(subset=[col])
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategies['handle_missing_categorical'] == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif strategies['handle_missing_categorical'] == 'remove':
                    df_clean = df_clean.dropna(subset=[col])
        
        # Convert appropriate columns to categorical
        if strategies['convert_to_categorical']:
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    unique_ratio = df_clean[col].nunique() / len(df_clean)
                    if unique_ratio < strategies['categorical_threshold']:
                        df_clean[col] = df_clean[col].astype('category')
                        messages.append(f"'{col}' column converted to categorical type")
        
        self.df = df_clean
        messages.append("Data cleaning completed!")
        return True, "\n".join(messages)
    
    def analyze_data(self):
        """Analyze the data"""
        if self.df is None:
            return None
        
        analysis_result = {}
        
        # Basic information
        analysis_result['shape'] = self.df.shape
        analysis_result['columns'] = self.df.columns.tolist()
        analysis_result['dtypes'] = self.df.dtypes.to_dict()
        analysis_result['memory_usage'] = self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_result['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_result['categorical_stats'] = {}
            for col in categorical_cols:
                analysis_result['categorical_stats'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_value': self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else None,
                    'top_frequency': self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
                }
        
        # Check for missing values
        analysis_result['missing_values'] = self.df.isnull().sum().to_dict()
        analysis_result['missing_percentage'] = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        
        # Identify trends in numeric data
        trends = {}
        for col in numeric_cols:
            skewness = self.df[col].skew()
            kurt = self.df[col].kurtosis()
            trends[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'skewness': skewness,
                'kurtosis': kurt,
                'trend': 'Increasing' if self.df[col].mean() > self.df[col].median() else 'Decreasing',
                'distribution': 'Highly skewed' if abs(skewness) > 1 else 'Moderately skewed' if abs(skewness) > 0.5 else 'Approximately symmetric'
            }
        
        analysis_result['trends'] = trends
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            analysis_result['correlation'] = self.df[numeric_cols].corr().to_dict()
        
        # Advanced statistics
        analysis_result['advanced_stats'] = {
            'outlier_info': self.detect_outliers(),
            'pairwise_correlations': self.get_top_correlations()
        }
        
        self.analysis_result = analysis_result
        return analysis_result
    
    def detect_outliers(self, threshold=1.5):
        """Detect outliers using IQR method"""
        if self.df is None:
            return {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outlier_info
    
    def get_top_correlations(self, n=5):
        """Get top positive and negative correlations"""
        if self.df is None:
            return {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[numeric_cols].corr()
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not pd.isna(corr_value):
                    correlations.append({
                        'columns': (col1, col2),
                        'correlation': corr_value
                    })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Return top n positive and negative correlations
        top_positive = [c for c in correlations if c['correlation'] > 0][:n]
        top_negative = [c for c in correlations if c['correlation'] < 0][:n]
        
        return {
            'top_positive': top_positive,
            'top_negative': top_negative
        }
    
    def display_analysis_results(self):
        """Display analysis results"""
        if not self.analysis_result:
            return "No analysis results available!"
        
        result = []
        result.append("="*60)
        result.append("Advanced Data Analysis Report")
        result.append("="*60)
        result.append(f"\nNumber of rows: {self.analysis_result['shape'][0]}")
        result.append(f"Number of columns: {self.analysis_result['shape'][1]}")
        result.append(f"Memory usage: {self.analysis_result['memory_usage']:.2f} MB")
        
        result.append("\nColumn information:")
        for col, dtype in self.analysis_result['dtypes'].items():
            result.append(f" {col}: {dtype}")
        
        result.append("\nMissing values:")
        for col, missing in self.analysis_result['missing_values'].items():
            if missing > 0:
                result.append(f" {col}: {missing} ({self.analysis_result['missing_percentage'][col]:.2f}%)")
        
        if 'categorical_stats' in self.analysis_result:
            result.append("\nCategorical data analysis:")
            for col, stats in self.analysis_result['categorical_stats'].items():
                result.append(f" {col}:")
                result.append(f"  Unique values: {stats['unique_count']}")
                result.append(f"  Most common value: {stats['top_value']} (frequency: {stats['top_frequency']})")
        
        if 'trends' in self.analysis_result:
            result.append("\nTrend and distribution analysis:")
            for col, stats in self.analysis_result['trends'].items():
                result.append(f" {col}:")
                result.append(f"  Mean: {stats['mean']:.2f}")
                result.append(f"  Median: {stats['median']:.2f}")
                result.append(f"  Standard deviation: {stats['std']:.2f}")
                result.append(f"  Skewness: {stats['skewness']:.2f} ({stats['distribution']})")
                result.append(f"  Kurtosis: {stats['kurtosis']:.2f}")
                result.append(f"  Trend: {stats['trend']}")
        
        # Display outlier information
        if 'advanced_stats' in self.analysis_result and 'outlier_info' in self.analysis_result['advanced_stats']:
            result.append("\nOutlier analysis:")
            for col, info in self.analysis_result['advanced_stats']['outlier_info'].items():
                if info['count'] > 0:
                    result.append(f" {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        # Display correlation information
        if 'advanced_stats' in self.analysis_result and 'pairwise_correlations' in self.analysis_result['advanced_stats']:
            correlations = self.analysis_result['advanced_stats']['pairwise_correlations']
            
            if 'top_positive' in correlations and correlations['top_positive']:
                result.append("\nTop positive correlations:")
                for corr in correlations['top_positive']:
                    cols = corr['columns']
                    result.append(f" {cols[0]} & {cols[1]}: {corr['correlation']:.3f}")
            
            if 'top_negative' in correlations and correlations['top_negative']:
                result.append("\nTop negative correlations:")
                for corr in correlations['top_negative']:
                    cols = corr['columns']
                    result.append(f" {cols[0]} & {cols[1]}: {corr['correlation']:.3f}")
        
        return "\n".join(result)
    
    def recommend_charts(self):
        """Recommend appropriate charts"""
        if self.df is None:
            return []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        recommendations = []
        
        # 1. Distribution charts
        if numeric_cols:
            rec = {
                'title': 'Distribution Charts',
                'charts': ['Histogram', 'Density Plot', 'Box Plot', 'Violin Plot'],
                'description': 'Ideal for showing the distribution of numerical data.',
                'reason': f'Your data has {len(numeric_cols)} numerical columns.',
                'columns': numeric_cols,
                'priority': 'High'
            }
            recommendations.append(rec)
        
        # 2. Composition charts
        if categorical_cols:
            rec = {
                'title': 'Composition Charts',
                'charts': ['Bar Chart', 'Pie Chart', 'Treemap', 'Waffle Chart'],
                'description': 'Perfect for comparing categories or showing composition.',
                'reason': f'Your data has {len(categorical_cols)} categorical columns.',
                'columns': categorical_cols,
                'priority': 'High'
            }
            recommendations.append(rec)
        
        # 3. Relationship charts
        if len(numeric_cols) >= 2:
            rec = {
                'title': 'Relationship Charts',
                'charts': ['Scatter Plot', 'Bubble Chart', 'Heatmap', 'Pairs Plot'],
                'description': 'Shows relationships and correlations between variables.',
                'reason': f'You have {len(numeric_cols)} numerical columns suitable for relationship analysis.',
                'columns': numeric_cols[:min(5, len(numeric_cols))],  # First 5 numeric columns
                'priority': 'Medium'
            }
            recommendations.append(rec)
        
        # 4. Comparison charts
        if numeric_cols and categorical_cols:
            rec = {
                'title': 'Comparison Charts',
                'charts': ['Grouped Bar Chart', 'Stacked Bar Chart', 'Box Plot', 'Violin Plot'],
                'description': 'Compares numerical values across categories.',
                'reason': 'You have both numerical and categorical data which is ideal for comparison.',
                'columns': [numeric_cols[0], categorical_cols[0]],  # First numeric and first categorical column
                'priority': 'Medium'
            }
            recommendations.append(rec)
        
        # 5. Trend charts
        if datetime_cols and numeric_cols:
            rec = {
                'title': 'Trend Charts',
                'charts': ['Line Chart', 'Area Chart', 'Stacked Area Chart'],
                'description': 'Shows trends and changes over time.',
                'reason': f'Your data has {len(datetime_cols)} datetime columns and {len(numeric_cols)} numerical columns.',
                'columns': [datetime_cols[0], numeric_cols[0]],  # First datetime and first numeric column
                'priority': 'High'
            }
            recommendations.append(rec)
        elif len(numeric_cols) >= 1 and self.df.shape[0] > 10:
            rec = {
                'title': 'Trend Charts',
                'charts': ['Line Chart', 'Area Chart'],
                'description': 'Shows trends in ordered data.',
                'reason': 'Your data has enough rows suitable for trend analysis.',
                'columns': numeric_cols[:1],  # First numeric column
                'priority': 'Medium'
            }
            recommendations.append(rec)
        
        # 6. Geospatial charts
        geo_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['country', 'city', 'state', 'latitude', 'longitude', 'location'])]
        if geo_cols and numeric_cols:
            rec = {
                'title': 'Geographical Charts',
                'charts': ['Choropleth Map', 'Scatter Geo Map', 'Bubble Map'],
                'description': 'Shows geographical data and location-based trends.',
                'reason': f'Your data has geographical columns: {geo_cols}',
                'columns': geo_cols[:1] + numeric_cols[:1],  # First geo and first numeric column
                'priority': 'Low'
            }
            recommendations.append(rec)
        
        return recommendations
    
    def perform_advanced_analysis(self):
        """Perform advanced analysis"""
        if self.df is None:
            return None, "No data loaded!"
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None, "Not enough numerical columns for advanced analysis!"
        
        # Prepare data for PCA
        df_numeric = self.df[numeric_cols].dropna()
        if len(df_numeric) < 10:
            return None, "Not enough data for PCA!"
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Perform clustering
        inertias = []
        k_range = range(2, min(8, len(df_numeric) // 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Choose optimal k (simplified method)
        optimal_k = 3  # Default
        if len(inertias) > 1:
            # Simple method to find the "elbow"
            differences = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            optimal_k = k_range[differences.index(max(differences)) + 1] if differences else 2
        
        # Apply KMeans with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to dataframe
        df_clustered = df_numeric.copy()
        df_clustered['Cluster'] = cluster_labels
        
        # Show cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_info = "\nCluster sizes:\n"
        for cluster, size in cluster_sizes.items():
            cluster_info += f"Cluster {cluster}: {size} samples ({size/len(cluster_labels):.1%})\n"
        
        return df_clustered, f"Suggested optimal number of clusters: {optimal_k}\nVariance explained by first 3 principal components: {np.cumsum(pca.explained_variance_ratio_)[2]:.2%}\n{cluster_info}"
    
    def generate_automated_report(self):
        """Generate automated analysis report"""
        if self.df is None:
            return False, "No data loaded!"
        
        try:
            report = sv.analyze(self.df)
            report_file = "SWEETVIZ_REPORT.html"
            report.show_html(filepath=report_file, open_browser=False)
            return True, report_file
        except Exception as e:
            return False, f"Error generating automated report: {str(e)}"
