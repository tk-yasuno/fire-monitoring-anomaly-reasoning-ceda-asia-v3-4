#!/usr/bin/env python3
"""
Global Fire Monitoring and Anomaly Reasoning System v3.3
NASA FIRMS + CEDA Fire_cci 統合監視システム + 異常検知 + MiniCPM推論

Enhanced Multi-Modal Fire Analysis with Anomaly Detection and AI Reasoning
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# システムパス設定
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

# v3.2システムからの継承用
try:
    # v3.2のコアシステムを利用
    sys.path.append(str(Path(__file__).parent.parent / 'global-fire-monitoring-v3-0'))
    from global_fire_monitoring_v32 import GlobalFireMonitoringSystemV32
    V32_AVAILABLE = True
except ImportError:
    V32_AVAILABLE = False
    print("⚠️ v3.2システムが利用できません")

class GlobalFireMonitoringAndAnomalyReasoningSystemV33:
    """
    Global Fire Monitoring and Anomaly Reasoning System v3.3
    
    v3.2の機能に加えて以下を追加：
    - Isolation Forest による異常グリッド検知
    - MiniCPM による異常理由推論
    - 異常グリッドの可視化と分析
    """
    
    def __init__(self, config_path='config/global_config_v33.json'):
        """
        初期化
        
        Args:
            config_path (str): 設定ファイルパス
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # v3.2システムの初期化
        if V32_AVAILABLE:
            try:
                self.v32_system = GlobalFireMonitoringSystemV32()
                self.logger.info("✅ v3.2システム初期化成功")
            except Exception as e:
                self.logger.warning(f"⚠️ v3.2システム初期化失敗: {e}")
                self.v32_system = None
        else:
            self.v32_system = None
            
        # 異常検知コンポーネント
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # MiniCPM推論エンジン（プレースホルダー）
        self.reasoning_engine = None
        
        # 結果保存用
        self.latest_results = {}
        
        self.logger.info("🔥 Global Fire Monitoring and Anomaly Reasoning System v3.3 初期化完了")
    
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fire_monitoring_anomaly_v33_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """設定ファイル読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"✅ 設定ファイル読み込み成功: {self.config_path}")
                return config
            else:
                self.logger.warning(f"⚠️ 設定ファイルが見つかりません: {self.config_path}")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"❌ 設定ファイル読み込みエラー: {e}")
            return self._create_default_config()
    
    def _create_default_config(self):
        """デフォルト設定の作成"""
        default_config = {
            "system": {
                "name": "Global Fire Monitoring and Anomaly Reasoning System",
                "version": "3.3",
                "description": "Enhanced fire monitoring with anomaly detection and AI reasoning"
            },
            "anomaly_detection": {
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42,
                "bootstrap": False
            },
            "reasoning": {
                "model": "minicpm",
                "max_tokens": 512,
                "temperature": 0.7
            },
            "data_processing": {
                "min_samples_per_continent": 2000,
                "total_samples": 12500,
                "feature_dimensions": 24
            },
            "visualization": {
                "anomaly_threshold": 0.1,
                "color_scheme": "viridis",
                "figure_size": [15, 10]
            }
        }
        
        # 設定ファイルを保存
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ デフォルト設定ファイル作成: {self.config_path}")
        return default_config
    
    def process_yearly_data_with_anomaly_detection(self, year=2022):
        """
        年次データ処理 + 異常検知
        
        Args:
            year (int): 処理年
            
        Returns:
            dict: 処理結果（通常データ + 異常検知結果）
        """
        self.logger.info(f"🔍 {year}年データの異常検知処理開始")
        
        # v3.2システムでベースデータ処理
        if self.v32_system:
            try:
                base_results = self.v32_system.process_yearly_data(year=year)
                self.logger.info("✅ v3.2ベースデータ処理完了")
            except Exception as e:
                self.logger.error(f"❌ v3.2データ処理エラー: {e}")
                # フォールバック：シミュレーションデータ
                base_results = self._generate_simulation_data()
        else:
            # v3.2が利用できない場合のシミュレーション
            base_results = self._generate_simulation_data()
        
        # 異常検知の実行
        anomaly_results = self._detect_anomalies(base_results)
        
        # 結果の統合
        combined_results = {
            'base_data': base_results,
            'anomaly_detection': anomaly_results,
            'processing_timestamp': datetime.now().isoformat(),
            'year': year
        }
        
        self.latest_results = combined_results
        return combined_results
    
    def _generate_simulation_data(self):
        """シミュレーションデータ生成（v3.2が利用できない場合）"""
        self.logger.info("🔄 シミュレーションデータ生成中...")
        
        # 12,500グリッドのシミュレーションデータ
        n_samples = 12500
        
        # 大陸ごとのサンプル数
        continent_samples = {
            'North America': 2100,
            'South America': 2000,
            'Africa': 2200,
            'Europe': 2000,
            'Asia': 2200,
            'Oceania': 2000
        }
        
        data_list = []
        
        for continent, n_cont_samples in continent_samples.items():
            # 大陸ごとの特徴パターン
            if continent == 'Africa':
                # アフリカ：高い火災活動
                base_fire_activity = np.random.gamma(3, 2, n_cont_samples)
                lat_range = (-35, 37)
                lon_range = (-20, 55)
            elif continent == 'North America':
                # 北米：中程度、季節変動大
                base_fire_activity = np.random.beta(2, 5, n_cont_samples) * 10
                lat_range = (25, 75)
                lon_range = (-170, -50)
            elif continent == 'Asia':
                # アジア：多様なパターン
                base_fire_activity = np.random.lognormal(1, 1.5, n_cont_samples)
                lat_range = (-10, 70)
                lon_range = (60, 180)
            else:
                # その他の大陸
                base_fire_activity = np.random.exponential(2, n_cont_samples)
                lat_range = (-50, 70)
                lon_range = (-180, 180)
            
            # 座標生成
            lats = np.random.uniform(lat_range[0], lat_range[1], n_cont_samples)
            lons = np.random.uniform(lon_range[0], lon_range[1], n_cont_samples)
            
            # 24次元特徴量生成
            for i in range(n_cont_samples):
                features = {
                    'latitude': lats[i],
                    'longitude': lons[i],
                    'continent': continent,
                    'fire_activity': base_fire_activity[i],
                    'burned_area_total': base_fire_activity[i] * np.random.uniform(0.5, 2.0),
                    'temperature_avg': np.random.normal(25, 10),
                    'precipitation_total': np.random.exponential(50),
                    'vegetation_index': np.random.beta(2, 3),
                    'elevation': abs(np.random.normal(500, 800)),
                    'slope': np.random.gamma(2, 2),
                    'distance_to_water': np.random.exponential(20),
                    'population_density': np.random.lognormal(2, 2),
                }
                
                # 近傍特徴量（重要度が高い）
                features.update({
                    'neighbor_max': features['fire_activity'] * np.random.uniform(1.2, 3.0),
                    'neighbor_std': features['fire_activity'] * np.random.uniform(0.2, 0.8),
                    'neighbor_mean': features['fire_activity'] * np.random.uniform(0.8, 1.5),
                    'neighbor_count': np.random.poisson(5),
                })
                
                # 月別パターン（12ヶ月）
                seasonal_pattern = np.sin(np.arange(12) * 2 * np.pi / 12) + 1
                for month in range(12):
                    features[f'month_{month+1}_fire'] = (
                        features['fire_activity'] * seasonal_pattern[month] * 
                        np.random.uniform(0.5, 1.5)
                    )
                
                data_list.append(features)
        
        df = pd.DataFrame(data_list)
        
        self.logger.info(f"✅ シミュレーションデータ生成完了: {len(df)}グリッド")
        return {'processed_data': df, 'metadata': {'data_type': 'simulation'}}
    
    def _detect_anomalies(self, base_results):
        """
        Isolation Forestによる異常検知
        
        Args:
            base_results (dict): v3.2からの基本処理結果
            
        Returns:
            dict: 異常検知結果
        """
        self.logger.info("🔍 Isolation Forest異常検知開始")
        
        try:
            # データの準備
            if 'processed_data' in base_results:
                df = base_results['processed_data']
            else:
                raise ValueError("処理済みデータが見つかりません")
            
            # 特徴量の選択（数値データのみ）
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            feature_data = df[numeric_columns].copy()
            
            # 欠損値の処理
            feature_data = feature_data.fillna(feature_data.median())
            
            self.logger.info(f"📊 特徴量数: {len(feature_data.columns)}")
            self.logger.info(f"📊 サンプル数: {len(feature_data)}")
            
            # データの標準化
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Isolation Forest設定
            anomaly_config = self.config['anomaly_detection']
            
            self.anomaly_detector = IsolationForest(
                contamination=anomaly_config['contamination'],
                n_estimators=anomaly_config['n_estimators'],
                random_state=anomaly_config['random_state'],
                bootstrap=anomaly_config['bootstrap'],
                n_jobs=-1
            )
            
            # 異常検知実行
            predictions = self.anomaly_detector.fit_predict(scaled_features)
            anomaly_scores = self.anomaly_detector.score_samples(scaled_features)
            
            # 結果の統合
            df['anomaly_prediction'] = predictions  # -1: 異常, 1: 正常
            df['anomaly_score'] = anomaly_scores    # スコア（低いほど異常）
            df['is_anomaly'] = (predictions == -1)
            
            # 異常グリッドの統計
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions)
            
            # 大陸別異常分布
            continent_anomalies = df.groupby('continent')['is_anomaly'].agg(['sum', 'count', 'mean'])
            
            # 特徴量重要度（異常グリッドの特徴分析）
            anomaly_df = df[df['is_anomaly']]
            normal_df = df[~df['is_anomaly']]
            
            feature_importance = {}
            for col in numeric_columns:
                if col in ['anomaly_prediction', 'anomaly_score', 'is_anomaly']:
                    continue
                
                try:
                    anomaly_mean = anomaly_df[col].mean()
                    normal_mean = normal_df[col].mean()
                    difference = abs(anomaly_mean - normal_mean) / (normal_mean + 1e-8)
                    feature_importance[col] = difference
                except:
                    feature_importance[col] = 0
            
            # 重要度でソート
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 異常グリッドのみを抽出
            anomaly_grids = df[df['is_anomaly']].copy()
            
            anomaly_results = {
                'detection_summary': {
                    'total_grids': len(predictions),
                    'anomalous_grids': n_anomalies,
                    'anomaly_rate': anomaly_rate,
                    'algorithm': 'isolation_forest',
                    'contamination_rate': anomaly_config['contamination']
                },
                'continent_distribution': continent_anomalies.to_dict(),
                'feature_importance': dict(sorted_features[:10]),  # 上位10特徴
                'anomaly_data': df,  # 全データ（異常フラグ付き）
                'anomaly_grids': anomaly_grids,  # 異常グリッドのみ
                'model_parameters': anomaly_config
            }
            
            self.logger.info(f"✅ 異常検知完了: {n_anomalies}個の異常グリッド検出 ({anomaly_rate:.1%})")
            return anomaly_results
            
        except Exception as e:
            self.logger.error(f"❌ 異常検知エラー: {e}")
            return {'error': str(e)}
    
    def analyze_anomalies_with_reasoning(self, top_n=10):
        """
        異常グリッドの詳細分析とMiniCPM推論
        
        Args:
            top_n (int): 分析する上位異常グリッド数
            
        Returns:
            dict: 推論結果
        """
        if not self.latest_results or 'anomaly_detection' not in self.latest_results:
            self.logger.error("❌ 異常検知結果が見つかりません")
            return {}
        
        self.logger.info(f"🤖 上位{top_n}異常グリッドの推論分析開始")
        
        try:
            anomaly_data = self.latest_results['anomaly_detection']['anomaly_data']
            anomaly_grids = anomaly_data[anomaly_data['is_anomaly']].copy()
            
            # 異常スコアで並べ替え（低いほど異常）
            anomaly_grids = anomaly_grids.sort_values('anomaly_score').head(top_n)
            
            reasoning_results = []
            
            for idx, row in anomaly_grids.iterrows():
                # 各異常グリッドの分析
                analysis = self._analyze_single_anomaly(row)
                
                # MiniCPM推論（プレースホルダー実装）
                reasoning = self._generate_anomaly_reasoning(row, analysis)
                
                result = {
                    'grid_id': idx,
                    'location': {
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'continent': row['continent']
                    },
                    'anomaly_score': row['anomaly_score'],
                    'analysis': analysis,
                    'reasoning': reasoning
                }
                
                reasoning_results.append(result)
            
            self.logger.info(f"✅ {len(reasoning_results)}個の異常グリッド推論完了")
            
            return {
                'reasoning_results': reasoning_results,
                'summary': self._summarize_anomaly_patterns(reasoning_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 異常推論エラー: {e}")
            return {'error': str(e)}
    
    def _analyze_single_anomaly(self, grid_data):
        """単一異常グリッドの分析"""
        analysis = {
            'key_features': {},
            'deviation_factors': [],
            'risk_level': 'unknown'
        }
        
        # 重要特徴量の分析
        important_features = ['fire_activity', 'burned_area_total', 'neighbor_max', 'neighbor_std']
        
        for feature in important_features:
            if feature in grid_data:
                value = grid_data[feature]
                analysis['key_features'][feature] = value
                
                # 偏差要因の判定
                if feature == 'fire_activity' and value > 10:
                    analysis['deviation_factors'].append('極端に高い火災活動')
                elif feature == 'neighbor_max' and value > 15:
                    analysis['deviation_factors'].append('近傍で異常な高火災値')
                elif feature == 'neighbor_std' and value > 5:
                    analysis['deviation_factors'].append('近傍火災の高い変動性')
        
        # リスクレベル判定
        fire_activity = grid_data.get('fire_activity', 0)
        if fire_activity > 15:
            analysis['risk_level'] = 'critical'
        elif fire_activity > 8:
            analysis['risk_level'] = 'high'
        elif fire_activity > 3:
            analysis['risk_level'] = 'moderate'
        else:
            analysis['risk_level'] = 'low'
        
        return analysis
    
    def _generate_anomaly_reasoning(self, grid_data, analysis):
        """
        MiniCPM推論の生成（プレースホルダー実装）
        
        実際の実装では、MiniCPMモデルを使用して推論を生成
        """
        # プレースホルダー推論ロジック
        continent = grid_data.get('continent', 'Unknown')
        fire_activity = grid_data.get('fire_activity', 0)
        risk_level = analysis['risk_level']
        deviation_factors = analysis['deviation_factors']
        
        # 基本的な推論テンプレート
        reasoning_templates = {
            'critical': f"{continent}の該当グリッドは極めて深刻な火災異常を示している。",
            'high': f"{continent}において高い火災リスクが検出された。",
            'moderate': f"{continent}で中程度の火災異常パターンが観測された。",
            'low': f"{continent}で軽微な火災パターンの異常が確認された。"
        }
        
        base_reasoning = reasoning_templates.get(risk_level, "異常パターンが検出された。")
        
        # 偏差要因の追加
        if deviation_factors:
            factors_text = "、".join(deviation_factors)
            detailed_reasoning = f"{base_reasoning} 主な要因: {factors_text}。"
        else:
            detailed_reasoning = f"{base_reasoning} 詳細な要因分析が必要。"
        
        # 推奨アクション
        if risk_level == 'critical':
            action = "即座の現地調査と緊急対応体制の確立を推奨。"
        elif risk_level == 'high':
            action = "早期の現地確認と監視強化を推奨。"
        else:
            action = "継続的な監視と定期的なフォローアップを推奨。"
        
        full_reasoning = f"{detailed_reasoning} {action}"
        
        return {
            'explanation': full_reasoning,
            'confidence': 0.75,  # プレースホルダー信頼度
            'model': 'minicpm_placeholder',
            'factors_identified': deviation_factors,
            'recommended_action': action
        }
    
    def _summarize_anomaly_patterns(self, reasoning_results):
        """異常パターンの要約"""
        if not reasoning_results:
            return {}
        
        # リスクレベル分布
        risk_levels = [r['analysis']['risk_level'] for r in reasoning_results]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        # 大陸分布
        continents = [r['location']['continent'] for r in reasoning_results]
        continent_distribution = {cont: continents.count(cont) for cont in set(continents)}
        
        # 共通要因
        all_factors = []
        for r in reasoning_results:
            all_factors.extend(r['analysis']['deviation_factors'])
        
        factor_frequency = {factor: all_factors.count(factor) for factor in set(all_factors)}
        
        return {
            'risk_level_distribution': risk_distribution,
            'continent_distribution': continent_distribution,
            'common_factors': dict(sorted(factor_frequency.items(), key=lambda x: x[1], reverse=True)),
            'total_anomalies_analyzed': len(reasoning_results)
        }
    
    def create_anomaly_visualization(self):
        """異常検知結果の可視化"""
        if not self.latest_results or 'anomaly_detection' not in self.latest_results:
            self.logger.error("❌ 可視化用データが見つかりません")
            return
        
        self.logger.info("📊 異常検知可視化作成開始")
        
        try:
            # 日本語フォント設定
            plt.rcParams['font.family'] = ['Yu Gothic', 'Hiragino Kaku Gothic Pro', 'Takao Gothic', 'Droid Sans Fallback', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # データの準備
            anomaly_data = self.latest_results['anomaly_detection']['anomaly_data']
            
            # 6パネル可視化
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Global Fire Monitoring and Anomaly Detection Analysis v3.3', fontsize=16, fontweight='bold')
            
            # 1. 異常スコア分布
            axes[0, 0].hist(anomaly_data['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(anomaly_data['anomaly_score'].quantile(0.1), color='red', linestyle='--', label='Anomaly Threshold')
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Anomaly Score Distribution')
            axes[0, 0].legend()
            
            # 2. 地理的分布（異常グリッド）
            normal_data = anomaly_data[~anomaly_data['is_anomaly']]
            anomaly_grids = anomaly_data[anomaly_data['is_anomaly']]
            
            axes[0, 1].scatter(normal_data['longitude'], normal_data['latitude'], 
                             c='lightblue', alpha=0.5, s=10, label='Normal')
            axes[0, 1].scatter(anomaly_grids['longitude'], anomaly_grids['latitude'], 
                             c='red', alpha=0.8, s=30, label='Anomaly')
            axes[0, 1].set_xlabel('Longitude')
            axes[0, 1].set_ylabel('Latitude')
            axes[0, 1].set_title('Geographic Distribution of Anomalies')
            axes[0, 1].legend()
            
            # 3. 大陸別異常率
            continent_stats = anomaly_data.groupby('continent')['is_anomaly'].agg(['sum', 'count'])
            continent_stats['anomaly_rate'] = continent_stats['sum'] / continent_stats['count']
            
            axes[0, 2].bar(range(len(continent_stats)), continent_stats['anomaly_rate'], 
                          color='orange', alpha=0.7)
            axes[0, 2].set_xticks(range(len(continent_stats)))
            axes[0, 2].set_xticklabels(continent_stats.index, rotation=45, ha='right')
            axes[0, 2].set_ylabel('Anomaly Rate')
            axes[0, 2].set_title('Anomaly Rate by Continent')
            
            # 4. 火災活動 vs 異常スコア
            axes[1, 0].scatter(anomaly_data['fire_activity'], anomaly_data['anomaly_score'], 
                             c=anomaly_data['is_anomaly'], cmap='RdYlBu', alpha=0.6)
            axes[1, 0].set_xlabel('Fire Activity')
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Fire Activity vs Anomaly Score')
            
            # 5. 特徴量重要度
            if 'feature_importance' in self.latest_results['anomaly_detection']:
                importance = self.latest_results['anomaly_detection']['feature_importance']
                features = list(importance.keys())[:8]  # 上位8特徴
                values = [importance[f] for f in features]
                
                axes[1, 1].barh(range(len(features)), values, color='green', alpha=0.7)
                axes[1, 1].set_yticks(range(len(features)))
                axes[1, 1].set_yticklabels(features)
                axes[1, 1].set_xlabel('Importance Score')
                axes[1, 1].set_title('Feature Importance for Anomaly Detection')
            
            # 6. 異常グリッド詳細統計
            axes[1, 2].text(0.1, 0.9, f"Total Grids: {len(anomaly_data):,}", transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.8, f"Anomalous Grids: {len(anomaly_grids):,}", transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.7, f"Anomaly Rate: {len(anomaly_grids)/len(anomaly_data):.1%}", transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.6, f"Algorithm: Isolation Forest", transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.5, f"Contamination: {self.config['anomaly_detection']['contamination']}", transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_title('Anomaly Detection Summary')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/anomaly_detection_analysis_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"✅ 可視化保存完了: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 可視化エラー: {e}")
    
    def run_complete_anomaly_analysis(self, year=2022, top_anomalies=10):
        """
        完全な異常検知分析パイプライン
        
        Args:
            year (int): 分析年
            top_anomalies (int): 詳細分析する異常グリッド数
            
        Returns:
            dict: 完全な分析結果
        """
        self.logger.info(f"🚀 Global Fire Monitoring and Anomaly Analysis v3.3 完全実行開始")
        
        try:
            # 1. データ処理 + 異常検知
            results = self.process_yearly_data_with_anomaly_detection(year=year)
            
            # 2. 異常グリッドの推論分析
            reasoning_results = self.analyze_anomalies_with_reasoning(top_n=top_anomalies)
            
            # 3. 可視化
            self.create_anomaly_visualization()
            
            # 4. 結果の統合
            complete_results = {
                'system_info': {
                    'name': 'Global Fire Monitoring and Anomaly Reasoning System',
                    'version': '3.3',
                    'processing_year': year,
                    'timestamp': datetime.now().isoformat()
                },
                'data_processing': results,
                'anomaly_reasoning': reasoning_results,
                'performance_metrics': {
                    'total_grids_processed': len(results['anomaly_detection']['anomaly_data']),
                    'anomalies_detected': results['anomaly_detection']['detection_summary']['anomalous_grids'],
                    'anomaly_rate': results['anomaly_detection']['detection_summary']['anomaly_rate'],
                    'reasoning_completed': len(reasoning_results.get('reasoning_results', []))
                }
            }
            
            # 結果保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/complete_anomaly_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ 完全分析結果保存: {output_file}")
            self.logger.info("🎉 Global Fire Monitoring and Anomaly Analysis v3.3 完了")
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"❌ 完全分析エラー: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # システムの実行例
    print("🔥 Global Fire Monitoring and Anomaly Reasoning System v3.3")
    print("=" * 60)
    
    # システム初期化
    system = GlobalFireMonitoringAndAnomalyReasoningSystemV33()
    
    # 完全分析実行
    results = system.run_complete_anomaly_analysis(year=2022, top_anomalies=15)
    
    if 'error' not in results:
        print(f"\n✅ 分析完了!")
        print(f"📊 処理グリッド数: {results['performance_metrics']['total_grids_processed']:,}")
        print(f"🚨 異常検知数: {results['performance_metrics']['anomalies_detected']:,}")
        print(f"📈 異常率: {results['performance_metrics']['anomaly_rate']:.1%}")
        print(f"🤖 推論完了数: {results['performance_metrics']['reasoning_completed']}")
    else:
        print(f"❌ エラー: {results['error']}")