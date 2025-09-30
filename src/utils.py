"""
Utility functions for Africa Fire Report Analysis Pipeline v2.0
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
        log_file (Optional[str]): Log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('africa_fire_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Set encoding to UTF-8 to handle Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    os.makedirs(directory, exist_ok=True)


def save_results(results: Dict[str, Any], output_path: str, format: str = 'json') -> None:
    """
    Save analysis results to file.
    
    Args:
        results (Dict[str, Any]): Analysis results
        output_path (str): Output file path
        format (str): Output format ('json', 'pickle', 'csv')
    """
    ensure_directory(os.path.dirname(output_path))
    
    if format.lower() == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_numpy_to_lists(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format.lower() == 'csv':
        if isinstance(results, dict) and 'data' in results:
            df = results['data']
            if isinstance(df, pd.DataFrame):
                df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load previously saved analysis results.
    
    Args:
        input_path (str): Input file path
        
    Returns:
        Dict[str, Any]: Loaded analysis results
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_ext == '.pkl' or file_ext == '.pickle':
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def convert_numpy_to_lists(obj: Any) -> Any:
    """
    Convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj (Any): Object to convert
        
    Returns:
        Any: Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj


def create_tsne_plot(embeddings: np.ndarray, labels: np.ndarray, 
                     output_path: str, title: str = "t-SNE Visualization") -> None:
    """
    Create t-SNE visualization of text clusters.
    
    Args:
        embeddings (np.ndarray): Text embeddings
        labels (np.ndarray): Cluster labels
        output_path (str): Output image path
        title (str): Plot title
    """
    from sklearn.manifold import TSNE
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color], label=f'Cluster {label}', alpha=0.7, s=50)
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_wordcloud(texts: List[str], output_path: str, **kwargs) -> None:
    """
    Generate word cloud from fire alert texts.
    
    Args:
        texts (List[str]): Input texts
        output_path (str): Output image path
        **kwargs: Word cloud customization options
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        # Fallback: create a simple word frequency plot
        create_word_frequency_plot(texts, output_path)
        return
    
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Default wordcloud parameters
    wordcloud_params = {
        'width': 800,
        'height': 400,
        'background_color': 'white',
        'colormap': 'viridis',
        'max_words': 100,
        'relative_scaling': 0.5,
        'min_font_size': 10
    }
    wordcloud_params.update(kwargs)
    
    # Generate word cloud
    wordcloud = WordCloud(**wordcloud_params).generate(combined_text)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Fire Alert Keywords')
    
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_word_frequency_plot(texts: List[str], output_path: str) -> None:
    """
    Create word frequency plot as fallback for wordcloud.
    
    Args:
        texts (List[str]): Input texts
        output_path (str): Output image path
    """
    from collections import Counter
    
    # Extract words
    all_words = []
    for text in texts:
        words = text.lower().split()
        # Filter out common stop words and short words
        words = [word for word in words if len(word) > 3 and 
                word not in ['fire', 'alert', 'detected', 'region']]
        all_words.extend(words)
    
    # Get top words
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(20)
    
    if not top_words:
        return
    
    # Create plot
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(words)), counts)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title('Most Frequent Words in Fire Alerts')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(count), va='center')
    
    plt.tight_layout()
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_regional_analysis_plot(data: pd.DataFrame, output_path: str) -> None:
    """
    Create regional analysis visualization.
    
    Args:
        data (pd.DataFrame): Fire alert data with regional information
        output_path (str): Output image path
    """
    if 'region' not in data.columns:
        return
    
    # Count alerts by region
    region_counts = data['region'].value_counts()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    region_counts.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Fire Alerts by African Region')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Number of Alerts')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(region_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Set3.colors)
    ax2.set_title('Regional Distribution of Fire Alerts')
    
    plt.tight_layout()
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_cluster_feature_plot(features: np.ndarray, labels: np.ndarray, 
                               feature_names: List[str], output_path: str) -> None:
    """
    Create cluster feature comparison plot.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Cluster labels
        feature_names (List[str]): Feature names
        output_path (str): Output image path
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_features = min(len(feature_names), features.shape[1], 10)  # Limit to 10 features
    
    # Calculate mean features per cluster
    cluster_means = []
    for label in unique_labels:
        mask = labels == label
        cluster_mean = np.mean(features[mask, :n_features], axis=0)
        cluster_means.append(cluster_mean)
    
    cluster_means = np.array(cluster_means)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize features for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_means = scaler.fit_transform(cluster_means)
    
    sns.heatmap(normalized_means, 
                xticklabels=feature_names[:n_features],
                yticklabels=[f'Cluster {i}' for i in unique_labels],
                cmap='RdYlBu_r', center=0, annot=True, fmt='.2f')
    
    plt.title('Cluster Feature Profiles (Standardized)')
    plt.xlabel('Features')
    plt.ylabel('Clusters')
    plt.tight_layout()
    
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering quality metrics.
    
    Args:
        embeddings (np.ndarray): Data embeddings
        labels (np.ndarray): Cluster labels
        
    Returns:
        Dict[str, float]: Clustering metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    metrics = {}
    
    try:
        # Silhouette score
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings, labels)
        else:
            metrics['silhouette_score'] = 0.0
        
        # Calinski-Harabasz score
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
        else:
            metrics['calinski_harabasz_score'] = 0.0
        
        # Davies-Bouldin score (lower is better)
        if len(np.unique(labels)) > 1:
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
        else:
            metrics['davies_bouldin_score'] = float('inf')
        
        # Number of clusters
        metrics['n_clusters'] = len(np.unique(labels))
        
        # Cluster size statistics
        cluster_sizes = [np.sum(labels == label) for label in np.unique(labels)]
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['std_cluster_size'] = np.std(cluster_sizes)
        metrics['min_cluster_size'] = np.min(cluster_sizes)
        metrics['max_cluster_size'] = np.max(cluster_sizes)
        
    except Exception as e:
        logging.warning(f"Error calculating clustering metrics: {e}")
        metrics = {
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'n_clusters': len(np.unique(labels)),
            'avg_cluster_size': 0.0,
            'std_cluster_size': 0.0,
            'min_cluster_size': 0,
            'max_cluster_size': 0
        }
    
    return metrics


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in human-readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict[str, float]: Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}


def create_summary_report(results: Dict[str, Any], output_path: str) -> None:
    """
    Create a summary report of analysis results.
    
    Args:
        results (Dict[str, Any]): Analysis results
        output_path (str): Output file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""# Africa Fire Report Analysis Pipeline v2.0 - Summary Report

**Generated:** {timestamp}

## Executive Summary

"""
    
    # Add key metrics if available
    if 'metadata' in results:
        metadata = results['metadata']
        report_content += f"- **Total Alerts Processed:** {metadata.get('total_alerts', 'N/A')}\n"
        report_content += f"- **Processing Time:** {format_execution_time(metadata.get('processing_time', 0))}\n"
        report_content += f"- **Countries Analyzed:** {len(metadata.get('countries', []))}\n\n"
    
    if 'clusters' in results:
        clusters = results['clusters']
        report_content += f"- **Clusters Identified:** {clusters.get('n_clusters', 'N/A')}\n"
        
        if 'cluster_info' in clusters:
            largest_cluster = max(clusters['cluster_info'].values(), 
                                key=lambda x: x.get('size', 0), default={})
            report_content += f"- **Largest Cluster Size:** {largest_cluster.get('size', 'N/A')}\n\n"
    
    # Add regional summary if available
    if 'regional_analysis' in results:
        report_content += "## Regional Analysis\n\n"
        for region, data in results['regional_analysis'].items():
            report_content += f"### {region}\n"
            report_content += f"- Total Alerts: {data.get('total_alerts', 'N/A')}\n"
            report_content += f"- Peak Activity: {data.get('peak_activity_date', 'N/A')}\n\n"
    
    # Save report
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)


def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return assessment.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        Dict[str, Any]: Data quality assessment
    """
    assessment = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': {},
        'data_types': {},
        'duplicates': 0,
        'quality_score': 0.0
    }
    
    # Check missing values
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_percent = (missing_count / len(data)) * 100
        assessment['missing_values'][col] = {
            'count': missing_count,
            'percentage': missing_percent
        }
    
    # Check data types
    assessment['data_types'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    # Check duplicates
    assessment['duplicates'] = data.duplicated().sum()
    
    # Calculate quality score (0-100)
    score = 100.0
    
    # Penalize for missing values
    avg_missing = np.mean([info['percentage'] for info in assessment['missing_values'].values()])
    score -= avg_missing
    
    # Penalize for duplicates
    duplicate_penalty = (assessment['duplicates'] / len(data)) * 100
    score -= duplicate_penalty
    
    assessment['quality_score'] = max(0.0, score)
    
    return assessment