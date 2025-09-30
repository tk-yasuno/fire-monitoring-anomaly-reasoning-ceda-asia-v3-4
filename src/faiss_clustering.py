#!/usr/bin/env python3
"""
FAISS-based K-means clustering for Africa Fire Report Analysis Pipeline v2.0
High-performance clustering using Facebook AI Similarity Search (FAISS)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional, Any, List
import time

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class FAISSKMeansClusterer:
    """
    FAISS-accelerated K-means clustering for fire alert text embeddings.
    
    Provides high-performance clustering using Facebook AI Similarity Search (FAISS)
    with token-based optimization for large-scale text analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize FAISS K-means clusterer.
        
        Args:
            logger: Logger instance for debugging and progress tracking
        """
        self.logger = logger or logging.getLogger(__name__)
        self.faiss_available = FAISS_AVAILABLE
        
        if not self.faiss_available:
            self.logger.warning("⚠️ FAISS not available. Falling back to standard clustering.")
        else:
            self.logger.info("✅ FAISS clustering module initialized")
    
    def optimize_embeddings_for_faiss(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Optimize embeddings for FAISS processing.
        
        Args:
            embeddings: Input embeddings (samples, features)
            
        Returns:
            Optimized embeddings for FAISS
        """
        if not self.faiss_available:
            return embeddings
        
        # Ensure float32 for FAISS compatibility
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for better clustering
        faiss.normalize_L2(embeddings)
        
        self.logger.info(f"📊 Optimized embeddings for FAISS: {embeddings.shape}, dtype: {embeddings.dtype}")
        return embeddings
    
    def estimate_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20, min_k: int = 2) -> int:
        """
        Estimate optimal number of clusters using FAISS K-means.
        
        Args:
            embeddings: Input embeddings
            max_k: Maximum number of clusters to test
            min_k: Minimum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if not self.faiss_available:
            return min(8, max_k)  # Fallback value
        
        self.logger.info(f"🔍 Estimating optimal clusters (k={min_k} to {max_k})...")
        
        n_samples, d = embeddings.shape
        max_k = min(max_k, n_samples // 5)  # Ensure reasonable cluster sizes
        
        best_k = min_k
        best_score = -1
        
        # Test different k values
        for k in range(min_k, max_k + 1):
            try:
                # FAISS K-means clustering
                kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, gpu=False)
                kmeans.train(embeddings)
                
                # Get cluster assignments
                _, labels = kmeans.index.search(embeddings, 1)
                labels = labels.flatten()
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    self.logger.info(f"  k={k}: silhouette={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
            except Exception as e:
                self.logger.warning(f"  k={k}: Failed - {e}")
                continue
        
        self.logger.info(f"🎯 Optimal clusters: k={best_k} (silhouette={best_score:.3f})")
        return best_k
    
    def faiss_kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int, 
                               niter: int = 50, gpu: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform FAISS K-means clustering.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            niter: Number of iterations
            gpu: Use GPU acceleration
            
        Returns:
            Tuple of (cluster labels, clustering info)
        """
        if not self.faiss_available:
            raise ImportError("FAISS not available for clustering")
        
        start_time = time.time()
        n_samples, d = embeddings.shape
        
        self.logger.info(f"🚀 Starting FAISS K-means clustering...")
        self.logger.info(f"   📊 Data: {n_samples} samples, {d} dimensions")
        self.logger.info(f"   🎯 Clusters: {n_clusters}")
        self.logger.info(f"   ⚡ GPU: {'Enabled' if gpu else 'Disabled'}")
        
        # Initialize FAISS K-means
        kmeans = faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=niter,
            verbose=True,
            spherical=False,
            gpu=gpu
        )
        
        # Train the model
        self.logger.info("🔄 Training FAISS K-means...")
        kmeans.train(embeddings)
        
        # Get cluster assignments
        self.logger.info("🏷️ Assigning cluster labels...")
        distances, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()
        distances = distances.flatten()
        
        elapsed_time = time.time() - start_time
        
        # Calculate clustering metrics
        metrics = self._calculate_clustering_metrics(embeddings, labels)
        
        clustering_info = {
            'algorithm': 'faiss_kmeans',
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'dimensions': d,
            'iterations': niter,
            'gpu_used': gpu,
            'training_time': elapsed_time,
            'centroids': kmeans.centroids.copy(),
            'inertia': float(np.sum(distances)),
            'metrics': metrics
        }
        
        self.logger.info(f"✅ FAISS K-means completed in {elapsed_time:.2f}s")
        self.logger.info(f"📊 Silhouette score: {metrics['silhouette_score']:.3f}")
        self.logger.info(f"📊 Inertia: {clustering_info['inertia']:.2f}")
        
        return labels, clustering_info
    
    def adaptive_faiss_clustering(self, embeddings: np.ndarray, 
                                 max_clusters: int = 20, min_clusters: int = 2,
                                 gpu: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptive FAISS clustering with automatic parameter selection.
        
        Args:
            embeddings: Input embeddings
            max_clusters: Maximum number of clusters
            min_clusters: Minimum number of clusters
            gpu: Use GPU acceleration
            
        Returns:
            Tuple of (cluster labels, clustering info)
        """
        if not self.faiss_available:
            raise ImportError("FAISS not available for adaptive clustering")
        
        # Optimize embeddings for FAISS
        optimized_embeddings = self.optimize_embeddings_for_faiss(embeddings)
        
        # Estimate optimal number of clusters
        optimal_k = self.estimate_optimal_clusters(optimized_embeddings, max_clusters, min_clusters)
        
        # Perform clustering with optimal k
        labels, clustering_info = self.faiss_kmeans_clustering(
            optimized_embeddings, 
            optimal_k, 
            niter=50, 
            gpu=gpu
        )
        
        clustering_info['optimal_k_estimation'] = True
        clustering_info['k_range_tested'] = (min_clusters, max_clusters)
        
        return labels, clustering_info
    
    def _calculate_clustering_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics = {}
        
        try:
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(embeddings, labels))
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, labels))
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
        except Exception as e:
            self.logger.warning(f"Failed to calculate metrics: {e}")
            metrics = {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
        
        return metrics
    
    def token_based_preprocessing(self, texts: List[str], max_tokens: int = 512) -> List[str]:
        """
        Token-based preprocessing for better FAISS performance.
        
        Args:
            texts: Input texts
            max_tokens: Maximum tokens per text
            
        Returns:
            Preprocessed texts
        """
        processed_texts = []
        
        for text in texts:
            # Simple token truncation (can be enhanced with proper tokenization)
            tokens = text.split()
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            processed_texts.append(' '.join(tokens))
        
        self.logger.info(f"📝 Token-based preprocessing: {len(texts)} texts, max_tokens={max_tokens}")
        return processed_texts
    
    def cluster_analysis_with_tokens(self, texts: List[str], labels: np.ndarray, 
                                   top_tokens: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Analyze clusters with token-based insights.
        
        Args:
            texts: Original texts
            labels: Cluster labels
            top_tokens: Number of top tokens per cluster
            
        Returns:
            Cluster analysis with token insights
        """
        cluster_analysis = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            # Token frequency analysis
            token_freq = {}
            for text in cluster_texts:
                for token in text.lower().split():
                    token_freq[token] = token_freq.get(token, 0) + 1
            
            # Get top tokens
            top_cluster_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:top_tokens]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                'top_tokens': top_cluster_tokens,
                'total_tokens': len(token_freq),
                'avg_text_length': np.mean([len(text.split()) for text in cluster_texts])
            }
        
        return cluster_analysis


def install_faiss_instructions():
    """
    Print installation instructions for FAISS.
    """
    print("""
🔧 FAISS Installation Instructions:

For CPU-only version:
    pip install faiss-cpu

For GPU version (CUDA required):
    pip install faiss-gpu

For conda users:
    conda install -c pytorch faiss-cpu
    # or
    conda install -c pytorch faiss-gpu

Note: GPU version requires NVIDIA CUDA toolkit.
""")


if __name__ == "__main__":
    # Example usage and testing
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available")
        install_faiss_instructions()
    else:
        print("✅ FAISS clustering module ready")
        
        # Example with random data
        np.random.seed(42)
        test_embeddings = np.random.randn(100, 50).astype(np.float32)
        test_texts = [f"Sample fire alert text {i}" for i in range(100)]
        
        clusterer = FAISSKMeansClusterer()
        labels, info = clusterer.adaptive_faiss_clustering(test_embeddings, max_clusters=10)
        
        print(f"🎯 Clustering completed: {info['n_clusters']} clusters")
        print(f"📊 Silhouette score: {info['metrics']['silhouette_score']:.3f}")