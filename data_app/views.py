import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Dataset
from .forms import DatasetUploadForm, AnalysisForm
import io
import base64
from scipy.stats import chi2_contingency

def upload_file(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            dataset.process_file()
            return redirect('analysis', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm()
    return render(request, 'data_app/upload.html', {'form': form})

def analyze_data(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    df = dataset.get_dataframe()
    
    form = AnalysisForm(columns=dataset.columns)
    
    if request.method == 'POST':
        form = AnalysisForm(request.POST, columns=dataset.columns)
        if form.is_valid():
            var1 = form.cleaned_data['variable_1']
            var2 = form.cleaned_data['variable_2']
            analysis_type = form.cleaned_data['analysis_type']
            
            if analysis_type == 'univariate':
                stats_data = calculate_univariate_stats(df, var1)
                plots = generate_univariate_plots(df, var1)
            else:
                stats_data = calculate_bivariate_stats(df, var1, var2)
                plots = generate_bivariate_plots(df, var1, var2)
                
            return JsonResponse({
                'stats': stats_data,
                'plots': plots
            })
    
    context = {
        'form': form,
        'dataset': dataset
    }
    return render(request, 'data_app/analysis.html', context)

def calculate_univariate_stats(df, variable):
    if df[variable].dtype in ['int64', 'float64']:
        stats = {
            'mean': float(df[variable].mean()),  # Convertir en float Python
            'median': float(df[variable].median()),
            'std': float(df[variable].std()),
            'min': float(df[variable].min()),
            'max': float(df[variable].max()),
            'skewness': float(df[variable].skew()),
            'kurtosis': float(df[variable].kurtosis())
        }
    else:
        value_counts = df[variable].value_counts()
        stats = {
            'mode': str(df[variable].mode()[0]),  # Convertir en str
            'unique_values': int(len(value_counts)),  # Convertir en int
            'frequencies': {str(k): int(v) for k, v in value_counts.items()}  # Convertir les clés et valeurs
        }
    return stats

def calculate_bivariate_stats(df, var1, var2):
    results = {}
    
    if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
        correlation = float(df[var1].corr(df[var2]))
        covariance = float(df[var1].cov(df[var2]))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[var1], df[var2])
        
        results.update({
            'correlation': correlation,
            'covariance': covariance,
            'regression': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'equation': f'y = {float(slope):.2f}x + {float(intercept):.2f}'
            }
        })
    
    elif df[var1].dtype == 'object' and df[var2].dtype == 'object':
        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results.update({
            'chi2_test': {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof)
            },
            'contingency_table': {
                str(k1): {str(k2): int(v) for k2, v in row.items()}
                for k1, row in contingency_table.to_dict().items()
            }
        })
    
    else:
        categorical_var = var1 if df[var1].dtype == 'object' else var2
        numerical_var = var2 if df[var1].dtype == 'object' else var1
        
        group_stats = df.groupby(categorical_var)[numerical_var].agg([
            'mean', 'median', 'std', 'count'
        ])
        
        # Convertir le DataFrame des statistiques groupées en dictionnaire avec des types natifs
        group_stats_dict = {
            str(k): {
                stat: float(v) if stat != 'count' else int(v)
                for stat, v in row.items()
            }
            for k, row in group_stats.to_dict('index').items()
        }
        
        categories = df[categorical_var].unique()
        f_stat, p_value = stats.f_oneway(*[df[df[categorical_var] == cat][numerical_var] 
                                         for cat in categories])
        
        results.update({
            'group_statistics': group_stats_dict,
            'anova_test': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value)
            }
        })
    
    return results

def generate_univariate_plots(df, variable):
    plots = {}
    plt.figure(figsize=(10, 6))
    
    if df[variable].dtype in ['int64', 'float64']:
        # Matplotlib plots
        plt.hist(df[variable], bins=30)
        plt.title(f'Histogram of {variable}')
        plots['matplotlib_histogram'] = get_plot_base64()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[variable])
        plt.title(f'Boxplot of {variable}')
        plots['seaborn_boxplot'] = get_plot_base64()
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df[variable])
        plt.title(f'Violin plot of {variable}')
        plots['seaborn_violin'] = get_plot_base64()
        
    else:
        # Pour les variables catégorielles
        value_counts = df[variable].value_counts()
        
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'Pie Chart of {variable}')
        plots['matplotlib_pie'] = get_plot_base64()
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=variable)
        plt.title(f'Count Plot of {variable}')
        plt.xticks(rotation=45)
        plots['seaborn_countplot'] = get_plot_base64()
    
    return plots

def generate_bivariate_plots(df, var1, var2):
    plt.close('all')
    plots = {}
    
    if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
        # Matplotlib scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df[var1], df[var2])
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f'Scatter Plot: {var1} vs {var2}')
        plots['matplotlib_scatter'] = get_plot_base64()
        
        # Seaborn regression plot
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=var1, y=var2)
        plt.title(f'Regression Plot: {var1} vs {var2}')
        plots['seaborn_regplot'] = get_plot_base64()
        
        # Seaborn jointplot
        g = sns.jointplot(data=df, x=var1, y=var2, kind='reg')
        plots['seaborn_jointplot'] = get_plot_base64()
        
        # Seaborn heatmap of correlation
        plt.figure(figsize=(8, 6))
        correlation_matrix = df[[var1, var2]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plots['seaborn_heatmap'] = get_plot_base64()
        
    elif df[var1].dtype == 'object' and df[var2].dtype == 'object':
        # Heatmap for categorical variables
        plt.figure(figsize=(10, 8))
        contingency_table = pd.crosstab(df[var1], df[var2])
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Contingency Table Heatmap: {var1} vs {var2}')
        plots['categorical_heatmap'] = get_plot_base64()
        
        # Grouped bar plot
        plt.figure(figsize=(12, 6))
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'Grouped Bar Plot: {var1} vs {var2}')
        plt.xticks(rotation=45)
        plots['grouped_barplot'] = get_plot_base64()
        
    else:
        # Pour une variable catégorielle et une variable numérique
        categorical_var = var1 if df[var1].dtype == 'object' else var2
        numerical_var = var2 if df[var1].dtype == 'object' else var1
        
        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=categorical_var, y=numerical_var)
        plt.title(f'Boxplot: {numerical_var} by {categorical_var}')
        plt.xticks(rotation=45)
        plots['seaborn_boxplot'] = get_plot_base64()
        
        # Violinplot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x=categorical_var, y=numerical_var)
        plt.title(f'Violin Plot: {numerical_var} by {categorical_var}')
        plt.xticks(rotation=45)
        plots['seaborn_violinplot'] = get_plot_base64()
        
        # Swarmplot
        plt.figure(figsize=(10, 6))
        sns.swarmplot(data=df, x=categorical_var, y=numerical_var)
        plt.title(f'Swarm Plot: {numerical_var} by {categorical_var}')
        plt.xticks(rotation=45)
        plots['seaborn_swarmplot'] = get_plot_base64()
    
    return plots

def get_plot_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    image_png = buffer.getvalue()
    plt.close('all')  # Close all figures
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    return graphic.decode('utf-8')