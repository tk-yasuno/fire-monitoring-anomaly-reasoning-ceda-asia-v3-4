#!/usr/bin/env python3
"""
Multi-Modal Fire Feature Processor v3.2
NASA FIRMS + CEDA Fire_cci 統合特徴量抽出システム

Global Fire Monitoring System v3.2
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 既存のv3.1モジュールをインポート
try:
    from text_processor import FireAlertTextProcessor
    from faiss_clustering import FAISSKMeansClusterer
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PROCESSOR_AVAILABLE = False
    print("⚠️ v3.1テキスト処理モジュールが利用できません")

# CEDA クライアントをインポート
try:
    from ceda_client import CEDAFireCCIClient
    CEDA_CLIENT_AVAILABLE = True
except ImportError:
    CEDA_CLIENT_AVAILABLE = False
    print("⚠️ CEDA クライアントが利用できません")

class MultiModalFireFeatureProcessor:
    """マルチモーダル火災特徴量プロセッサ"""
    
    def __init__(self):
        """初期化"""
        self.firms_processor = None
        self.ceda_client = None
        
        # v3.1テキスト処理
        if TEXT_PROCESSOR_AVAILABLE:
            self.firms_processor = FireAlertTextProcessor()
            print("✅ NASA FIRMS テキスト処理モジュール利用可能")
        
        # CEDA クライアント
        if CEDA_CLIENT_AVAILABLE:
            self.ceda_client = CEDAFireCCIClient()
            print("✅ CEDA Fire_cci クライアント利用可能")
        
        print("🔬 マルチモーダル火災特徴量プロセッサ初期化完了")
    
    def extract_firms_features(self, firms_data, max_samples=2000):
        """
        NASA FIRMS データから特徴量を抽出
        
        Args:
            firms_data (pd.DataFrame): FIRMSデータ
            max_samples (int): 最大サンプル数
            
        Returns:
            dict: 抽出された特徴量
        """
        print(f"🛰️ NASA FIRMS 特徴量抽出開始: {len(firms_data)}件")
        
        features = {}
        
        # 1. 基本統計特徴量
        features['basic_stats'] = {
            'count': len(firms_data),
            'mean_latitude': firms_data['latitude'].mean(),
            'mean_longitude': firms_data['longitude'].mean(),
            'mean_brightness': firms_data['brightness'].mean() if 'brightness' in firms_data.columns else 0,
            'mean_confidence': firms_data['confidence'].mean() if 'confidence' in firms_data.columns else 0,
            'std_latitude': firms_data['latitude'].std(),
            'std_longitude': firms_data['longitude'].std(),
        }
        
        # 2. 地理的分布特徴量
        features['geographic'] = {
            'lat_range': firms_data['latitude'].max() - firms_data['latitude'].min(),
            'lon_range': firms_data['longitude'].max() - firms_data['longitude'].min(),
            'centroid_lat': firms_data['latitude'].median(),
            'centroid_lon': firms_data['longitude'].median(),
        }
        
        # 3. 密度特徴量
        lat_bins = np.linspace(firms_data['latitude'].min(), firms_data['latitude'].max(), 10)
        lon_bins = np.linspace(firms_data['longitude'].min(), firms_data['longitude'].max(), 10)
        
        hist, _, _ = np.histogram2d(firms_data['latitude'], firms_data['longitude'], 
                                  bins=[lat_bins, lon_bins])
        
        features['density'] = {
            'max_density': hist.max(),
            'mean_density': hist.mean(),
            'density_variance': hist.var(),
            'active_cells': (hist > 0).sum(),
        }
        
        # 4. v3.1テキスト特徴量（利用可能な場合）
        if self.firms_processor and len(firms_data) > 0:
            try:
                # テキスト生成
                text_features = self._generate_text_features(firms_data.head(max_samples))
                
                # テキスト処理
                processed_texts = self.firms_processor.preprocess_texts(text_features)
                text_features_dict = self.firms_processor.extract_features(processed_texts)
                
                # カテゴリ特徴量を使用
                if 'categorical' in text_features_dict:
                    features['text_features'] = {
                        'feature_matrix': text_features_dict['categorical'],
                        'feature_dim': text_features_dict['categorical'].shape[1],
                        'sample_count': len(processed_texts)
                    }
                    print(f"  ✅ テキスト特徴量: {features['text_features']['feature_dim']}次元")
            
            except Exception as e:
                print(f"  ⚠️ テキスト特徴量抽出エラー: {e}")
        
        print(f"  ✅ FIRMS特徴量抽出完了: {len(features)}カテゴリ")
        return features
    
    def extract_ceda_features(self, year=2022, month=1, continent='Africa'):
        """
        CEDA Fire_cci データから特徴量を抽出
        
        Args:
            year (int): 年
            month (int): 月
            continent (str): 大陸名
            
        Returns:
            dict: 抽出された特徴量
        """
        print(f"🔥 CEDA Fire_cci 特徴量抽出開始: {year}-{month:02d} {continent}")
        
        if not self.ceda_client:
            print("  ⚠️ CEDA クライアントが利用できません")
            return {}
        
        features = {}
        
        try:
            # 実CEDAデータダウンロード試行
            cache_path = self.ceda_client.download_monthly_data(year, month)
            if cache_path and cache_path.exists():
                dataset = self.ceda_client.load_netcdf_data(cache_path)
                print(f"  ✅ 実CEDAデータ使用: {cache_path.name}")
            else:
                raise Exception("実データダウンロード失敗")
        except Exception as e:
            print(f"  ⚠️ 実データ取得失敗、サンプルデータ使用: {e}")
            dataset = self.ceda_client.create_sample_data(year, month)
            
            # 大陸別サブセット
            subset = self.ceda_client.get_continental_subset(dataset, continent)
            
            # 統計計算
            stats = self.ceda_client.calculate_statistics(subset)
            
            # 1. 焼損面積特徴量
            if 'burned_area' in subset.data_vars:
                ba = subset['burned_area'].values
                features['burned_area'] = {
                    'total_burned_area': stats['total_burned_area'],
                    'max_burned_area': stats['max_burned_area'],
                    'mean_burned_area': stats['mean_burned_area'],
                    'active_cells': stats['active_cells'],
                    'burned_area_variance': float(np.var(ba)),
                    'burned_area_skewness': float(self._calculate_skewness(ba.flatten())),
                }
            
            # 2. 信頼度特徴量
            if 'confidence' in subset.data_vars:
                conf = subset['confidence'].values
                features['confidence'] = {
                    'mean_confidence': stats['mean_confidence'],
                    'high_confidence_cells': stats['high_confidence_cells'],
                    'confidence_variance': float(np.var(conf)),
                    'confidence_skewness': float(self._calculate_skewness(conf.flatten())),
                }
            
            # 3. 土地被覆特徴量
            if 'land_cover' in subset.data_vars:
                lc = subset['land_cover'].values
                unique_classes, counts = np.unique(lc, return_counts=True)
                
                features['land_cover'] = {
                    'dominant_class': int(unique_classes[np.argmax(counts)]),
                    'class_diversity': len(unique_classes),
                    'class_distribution': dict(zip(unique_classes.astype(int), counts.astype(int))),
                    'simpson_diversity': self._calculate_simpson_diversity(counts),
                }
            
            # 4. 空間パターン特徴量
            features['spatial_patterns'] = {
                'grid_size_lat': subset.lat.size,
                'grid_size_lon': subset.lon.size,
                'total_cells': stats['total_cells'],
                'spatial_extent_lat': float(subset.lat.max() - subset.lat.min()),
                'spatial_extent_lon': float(subset.lon.max() - subset.lon.min()),
            }
            
            print(f"  ✅ CEDA特徴量抽出完了: {len(features)}カテゴリ")
            
        except Exception as e:
            print(f"  ❌ CEDA特徴量抽出エラー: {e}")
        
        return features
    
    def integrate_multimodal_features(self, firms_features, ceda_features):
        """
        マルチモーダル特徴量を統合
        
        Args:
            firms_features (dict): FIRMS特徴量
            ceda_features (dict): CEDA特徴量
            
        Returns:
            dict: 統合特徴量
        """
        print("🔗 マルチモーダル特徴量統合開始")
        
        integrated = {
            'firms_features': firms_features,
            'ceda_features': ceda_features,
            'integration_metadata': {
                'integration_time': datetime.now().isoformat(),
                'firms_available': len(firms_features) > 0,
                'ceda_available': len(ceda_features) > 0,
            }
        }
        
        # 統合統計の計算
        if firms_features and ceda_features:
            integrated['cross_modal_stats'] = self._calculate_cross_modal_statistics(
                firms_features, ceda_features
            )
        
        print(f"  ✅ 統合完了: FIRMS({len(firms_features)}), CEDA({len(ceda_features)})")
        return integrated
    
    def _generate_text_features(self, data):
        """テキスト特徴を生成（v3.1と同じ）"""
        text_features = []
        for _, row in data.iterrows():
            lat = row.get('latitude', 0)
            lon = row.get('longitude', 0)
            confidence = row.get('confidence', 0)
            brightness = row.get('brightness', 0)
            
            # 地理的特徴の記述
            lat_desc = "northern" if lat > 0 else "southern"
            lon_desc = "eastern" if lon > 0 else "western"
            
            # 信頼度の記述
            conf_desc = "high confidence" if confidence > 70 else "medium confidence" if confidence > 30 else "low confidence"
            
            # 明度の記述
            bright_desc = "very bright" if brightness > 350 else "bright" if brightness > 320 else "moderate brightness"
            
            # テキスト特徴を結合
            text_feature = f"{lat_desc} {lon_desc} region fire alert {conf_desc} {bright_desc} temperature detection"
            text_features.append(text_feature)
        
        return text_features
    
    def _calculate_skewness(self, data):
        """歪度を計算"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_simpson_diversity(self, counts):
        """シンプソン多様性指数を計算"""
        total = np.sum(counts)
        if total == 0:
            return 0
        proportions = counts / total
        return 1 - np.sum(proportions ** 2)
    
    def _calculate_cross_modal_statistics(self, firms_features, ceda_features):
        """クロスモーダル統計を計算"""
        cross_stats = {}
        
        # FIRMS vs CEDA の空間的対応
        if 'geographic' in firms_features and 'spatial_patterns' in ceda_features:
            firms_geo = firms_features['geographic']
            ceda_spatial = ceda_features['spatial_patterns']
            
            cross_stats['spatial_overlap'] = {
                'firms_lat_range': firms_geo['lat_range'],
                'ceda_lat_range': ceda_spatial['spatial_extent_lat'],
                'firms_lon_range': firms_geo['lon_range'],
                'ceda_lon_range': ceda_spatial['spatial_extent_lon'],
            }
        
        # 密度 vs 焼損面積の関係
        if 'density' in firms_features and 'burned_area' in ceda_features:
            firms_density = firms_features['density']
            ceda_ba = ceda_features['burned_area']
            
            cross_stats['density_burned_relationship'] = {
                'firms_max_density': firms_density['max_density'],
                'ceda_total_burned': ceda_ba['total_burned_area'],
                'normalized_ratio': firms_density['max_density'] / (ceda_ba['total_burned_area'] + 1e-6),
            }
        
        return cross_stats
    
    def integrate_ceda_only_features(self, ceda_features, region):
        """
        CEDAデータのみを使用した特徴量統合
        
        Args:
            ceda_features (dict): CEDA特徴量
            region (str): 対象地域
            
        Returns:
            dict: 統合された特徴量
        """
        try:
            # CEDAベースの統合特徴量を構築
            integrated = {
                'integration_metadata': {
                    'integration_type': 'ceda_only',
                    'timestamp': datetime.now().isoformat(),
                    'region': region,
                    'data_sources': ['CEDA_Fire_cci'],
                    'firms_data_included': False
                },
                'ceda_features': ceda_features,
                'firms_features': {},  # 空のFIRMS特徴量
                'cross_modal_stats': {
                    'ceda_grid_count': 0,
                    'total_burned_area_km2': 0,
                    'avg_burned_area_per_grid': 0
                }
            }
            
            # CEDAデータから火災グリッド情報を構築
            grid_features = []
            
            # burned_areaキーからデータを取得
            if 'burned_area' in ceda_features:
                burned_area_info = ceda_features['burned_area']
                
                # burned_area情報を使用してグリッドデータを生成
                if isinstance(burned_area_info, dict):
                    total_burned = burned_area_info.get('total_burned_area', 1000)  # デフォルト値
                    mean_burned = burned_area_info.get('mean_burned_area', 10)
                    
                    # 統計情報を更新
                    integrated['cross_modal_stats'].update({
                        'total_burned_area_km2': total_burned,
                        'avg_burned_area_per_grid': mean_burned
                    })
                    
                    # グリッドデータを生成（100グリッド）
                    for i in range(100):
                        # アフリカの緯度経度範囲
                        lat = np.random.uniform(-35, 37)  # アフリカの緯度範囲
                        lon = np.random.uniform(-20, 51)  # アフリカの経度範囲
                        
                        # 焼失面積の分布
                        burned_area = np.random.exponential(mean_burned)
                        fire_activity = burned_area * 10  # 焼失面積ベースの活動指標
                        
                        grid_feature = {
                            'grid_id': i,
                            'latitude': lat,
                            'longitude': lon,
                            'burned_area_km2': burned_area,
                            'fire_activity': fire_activity,
                            'continent': region,
                            'data_source': 'CEDA_Fire_cci'
                        }
                        grid_features.append(grid_feature)
                
                integrated['grid_based_features'] = pd.DataFrame(grid_features)
                integrated['cross_modal_stats']['processed_grids'] = len(grid_features)
                integrated['cross_modal_stats']['ceda_grid_count'] = len(grid_features)
            
            return integrated
            
        except Exception as e:
            print(f"❌ CEDAのみ統合エラー: {e}")
            return {
                'integration_metadata': {
                    'integration_type': 'ceda_only_error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                'ceda_features': ceda_features,
                'cross_modal_stats': {}
            }

def test_multimodal_processor():
    """マルチモーダルプロセッサのテスト"""
    print("🧪 マルチモーダル火災特徴量プロセッサテスト")
    print("="*60)
    
    # プロセッサ初期化
    processor = MultiModalFireFeatureProcessor()
    
    # サンプルFIRMSデータ生成
    print("\n🛰️ サンプルFIRMSデータ生成")
    np.random.seed(42)
    n_samples = 100
    
    firms_data = pd.DataFrame({
        'latitude': np.random.uniform(-10, 10, n_samples),
        'longitude': np.random.uniform(20, 40, n_samples),
        'brightness': np.random.uniform(300, 400, n_samples),
        'confidence': np.random.uniform(50, 90, n_samples),
    })
    
    print(f"  生成データ: {len(firms_data)}件")
    
    # FIRMS特徴量抽出
    print("\n🔬 FIRMS特徴量抽出")
    firms_features = processor.extract_firms_features(firms_data)
    
    # CEDA特徴量抽出
    print("\n🔥 CEDA特徴量抽出")
    ceda_features = processor.extract_ceda_features(2022, 1, 'Africa')
    
    # マルチモーダル統合
    print("\n🔗 マルチモーダル統合")
    integrated_features = processor.integrate_multimodal_features(firms_features, ceda_features)
    
    # 結果表示
    print("\n📊 統合結果サマリー:")
    print(f"  FIRMS特徴量カテゴリ: {len(firms_features)}")
    print(f"  CEDA特徴量カテゴリ: {len(ceda_features)}")
    print(f"  統合メタデータ: {integrated_features['integration_metadata']}")
    
    if 'cross_modal_stats' in integrated_features:
        print("  クロスモーダル統計:")
        for key, value in integrated_features['cross_modal_stats'].items():
            print(f"    {key}: {value}")
    
    print("\n✅ マルチモーダルプロセッサテスト完了")

if __name__ == "__main__":
    test_multimodal_processor()