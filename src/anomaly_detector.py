#!/usr/bin/env python3
"""
Isolation Forest異常検知モジュール for Global Fire Monitoring System v3.3
グリッド単位での火災異常パターン検知
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FireAnomalyDetector:
    """
    Isolation Forest based 火災異常検知システム
    """
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config (dict): 異常検知設定
        """
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.config = config or {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'bootstrap': False,
            'n_jobs': -1,
            'max_features': 1.0
        }
        
        # コンポーネント
        self.detector = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        # 結果保存
        self.detection_results = {}
        self.feature_importance = {}
        
        self.logger.info("🔍 Fire Anomaly Detector初期化完了")
    
    def prepare_features(self, data_df, feature_selection='auto'):
        """
        特徴量準備と前処理
        
        Args:
            data_df (pd.DataFrame): 入力データ
            feature_selection (str): 特徴選択方法
            
        Returns:
            np.ndarray: 前処理済み特徴量
        """
        self.logger.info("📊 特徴量準備開始")
        
        try:
            # 数値列の自動選択
            if feature_selection == 'auto':
                numeric_columns = data_df.select_dtypes(include=[np.number]).columns.tolist()
                # 除外すべき列
                exclude_columns = ['anomaly_prediction', 'anomaly_score', 'is_anomaly']
                numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            else:
                numeric_columns = feature_selection
            
            self.feature_names = numeric_columns
            
            # 特徴量データ抽出
            feature_data = data_df[numeric_columns].copy()
            
            # 欠損値処理
            initial_shape = feature_data.shape
            feature_data = feature_data.fillna(feature_data.median())
            
            self.logger.info(f"📊 特徴量: {len(numeric_columns)}個")
            self.logger.info(f"📊 サンプル: {len(feature_data)}個")
            self.logger.info(f"📊 欠損値処理: {initial_shape} → {feature_data.shape}")
            
            # 特徴量統計
            self._log_feature_statistics(feature_data)
            
            return feature_data.values
            
        except Exception as e:
            self.logger.error(f"❌ 特徴量準備エラー: {e}")
            raise
    
    def _log_feature_statistics(self, feature_data):
        """特徴量統計のログ出力"""
        stats = feature_data.describe()
        
        self.logger.info("📈 特徴量統計:")
        for col in feature_data.columns[:5]:  # 最初の5特徴量のみ表示
            mean_val = stats.loc['mean', col]
            std_val = stats.loc['std', col]
            self.logger.info(f"  {col}: 平均={mean_val:.3f}, 標準偏差={std_val:.3f}")
    
    def configure_detector(self, optimization=False, cv_folds=3):
        """
        Isolation Forest検知器の設定と最適化
        
        Args:
            optimization (bool): ハイパーパラメータ最適化を実行するか
            cv_folds (int): 交差検証のフォールド数
        """
        self.logger.info("⚙️ Isolation Forest設定開始")
        
        if optimization:
            self.logger.info("🔧 ハイパーパラメータ最適化実行中...")
            
            # パラメータグリッド
            param_grid = {
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 100, 200],
                'max_features': [0.5, 0.7, 1.0],
                'bootstrap': [False, True]
            }
            
            # 基本検知器
            base_detector = IsolationForest(
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            )
            
            # グリッドサーチ（カスタム評価関数）
            grid_search = GridSearchCV(
                base_detector,
                param_grid,
                cv=cv_folds,
                scoring=self._custom_anomaly_scorer,
                n_jobs=1  # Isolation Forestが並列処理を行うため
            )
            
            # 最適化実行（ダミーターゲット）
            # Isolation Forestは教師なし学習のため、ダミーターゲットを使用
            dummy_target = np.zeros(len(self.current_features))
            grid_search.fit(self.current_features, dummy_target)
            
            # 最適パラメータ
            best_params = grid_search.best_params_
            self.logger.info(f"✅ 最適パラメータ: {best_params}")
            
            # 最適検知器の設定
            self.detector = IsolationForest(**best_params, random_state=self.config['random_state'])
            
        else:
            # デフォルト設定
            self.detector = IsolationForest(
                contamination=self.config['contamination'],
                n_estimators=self.config['n_estimators'],
                random_state=self.config['random_state'],
                bootstrap=self.config['bootstrap'],
                n_jobs=self.config['n_jobs'],
                max_features=self.config['max_features']
            )
        
        self.logger.info("✅ Isolation Forest設定完了")
    
    def _custom_anomaly_scorer(self, estimator, X, y=None):
        """カスタム異常検知評価関数"""
        try:
            # 異常予測
            predictions = estimator.fit_predict(X)
            anomaly_scores = estimator.score_samples(X)
            
            # 評価指標の組み合わせ
            # 1. 異常率が目標範囲内かどうか
            anomaly_rate = (predictions == -1).mean()
            rate_score = 1.0 - abs(anomaly_rate - 0.1) * 5  # 10%を目標
            
            # 2. 異常スコアの分離度
            normal_scores = anomaly_scores[predictions == 1]
            anomaly_scores_subset = anomaly_scores[predictions == -1]
            
            if len(anomaly_scores_subset) > 0 and len(normal_scores) > 0:
                separation = abs(np.mean(normal_scores) - np.mean(anomaly_scores_subset))
            else:
                separation = 0
            
            # 総合スコア
            total_score = rate_score * 0.6 + min(1.0, separation) * 0.4
            
            return total_score
            
        except Exception:
            return 0.0
    
    def detect_anomalies(self, feature_data, scale_data=True):
        """
        異常検知実行
        
        Args:
            feature_data (np.ndarray): 特徴量データ
            scale_data (bool): データスケーリングを行うか
            
        Returns:
            dict: 検知結果
        """
        self.logger.info("🔍 異常検知開始")
        
        try:
            # データスケーリング
            if scale_data:
                self.scaler = RobustScaler()  # 外れ値に頑健
                scaled_features = self.scaler.fit_transform(feature_data)
                self.logger.info("📏 RobustScaler適用")
            else:
                scaled_features = feature_data
            
            self.current_features = scaled_features
            
            # 検知器設定
            if self.detector is None:
                self.configure_detector()
            
            # 異常検知実行
            start_time = datetime.now()
            
            predictions = self.detector.fit_predict(scaled_features)
            anomaly_scores = self.detector.score_samples(scaled_features)
            
            detection_time = (datetime.now() - start_time).total_seconds()
            
            # 結果集約
            results = self._compile_detection_results(
                predictions, anomaly_scores, feature_data, detection_time
            )
            
            self.detection_results = results
            self.is_fitted = True
            
            self.logger.info(f"✅ 異常検知完了: {results['summary']['anomalous_count']}個検出")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 異常検知エラー: {e}")
            raise
    
    def _compile_detection_results(self, predictions, anomaly_scores, original_features, detection_time):
        """検知結果の集約"""
        
        # 基本統計
        total_samples = len(predictions)
        anomalous_count = (predictions == -1).sum()
        anomaly_rate = anomalous_count / total_samples
        
        # スコア統計
        score_stats = {
            'mean': np.mean(anomaly_scores),
            'std': np.std(anomaly_scores),
            'min': np.min(anomaly_scores),
            'max': np.max(anomaly_scores),
            'median': np.median(anomaly_scores)
        }
        
        # 異常・正常グループの分離度
        normal_scores = anomaly_scores[predictions == 1]
        anomaly_scores_subset = anomaly_scores[predictions == -1]
        
        separation_metrics = {}
        if len(anomaly_scores_subset) > 0 and len(normal_scores) > 0:
            separation_metrics = {
                'normal_mean': np.mean(normal_scores),
                'anomaly_mean': np.mean(anomaly_scores_subset),
                'separation_distance': abs(np.mean(normal_scores) - np.mean(anomaly_scores_subset)),
                'effect_size': abs(np.mean(normal_scores) - np.mean(anomaly_scores_subset)) / np.std(anomaly_scores)
            }
        
        # 特徴量重要度分析
        feature_importance = self._analyze_feature_importance(original_features, predictions)
        
        results = {
            'summary': {
                'total_samples': total_samples,
                'anomalous_count': anomalous_count,
                'anomaly_rate': anomaly_rate,
                'detection_time': detection_time,
                'algorithm': 'isolation_forest'
            },
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'score_statistics': score_stats,
            'separation_metrics': separation_metrics,
            'feature_importance': feature_importance,
            'model_parameters': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _analyze_feature_importance(self, original_features, predictions):
        """特徴量重要度分析"""
        if self.feature_names is None:
            return {}
        
        importance_scores = {}
        
        # 各特徴量について異常・正常グループ間の差を計算
        for i, feature_name in enumerate(self.feature_names):
            feature_values = original_features[:, i]
            
            normal_values = feature_values[predictions == 1]
            anomaly_values = feature_values[predictions == -1]
            
            if len(anomaly_values) > 0 and len(normal_values) > 0:
                # 平均値の差
                mean_diff = abs(np.mean(anomaly_values) - np.mean(normal_values))
                
                # 標準化された差（Cohen's d）
                pooled_std = np.sqrt(
                    ((len(normal_values) - 1) * np.var(normal_values) + 
                     (len(anomaly_values) - 1) * np.var(anomaly_values)) / 
                    (len(normal_values) + len(anomaly_values) - 2)
                )
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                else:
                    cohens_d = 0
                
                # 分布の重複度
                overlap = self._calculate_distribution_overlap(normal_values, anomaly_values)
                
                # 総合重要度（0-100スケール）
                importance = min(100, cohens_d * 20 * (1 - overlap))
                
                importance_scores[feature_name] = {
                    'importance_score': importance,
                    'mean_difference': mean_diff,
                    'cohens_d': cohens_d,
                    'distribution_overlap': overlap,
                    'normal_mean': np.mean(normal_values),
                    'anomaly_mean': np.mean(anomaly_values)
                }
        
        # 重要度でソート
        sorted_importance = dict(
            sorted(importance_scores.items(), 
                  key=lambda x: x[1]['importance_score'], 
                  reverse=True)
        )
        
        self.feature_importance = sorted_importance
        
        return sorted_importance
    
    def _calculate_distribution_overlap(self, values1, values2):
        """2つの分布の重複度計算"""
        try:
            # ヒストグラムベースの重複計算
            min_val = min(np.min(values1), np.min(values2))
            max_val = max(np.max(values1), np.max(values2))
            
            bins = np.linspace(min_val, max_val, 50)
            
            hist1, _ = np.histogram(values1, bins=bins, density=True)
            hist2, _ = np.histogram(values2, bins=bins, density=True)
            
            # 重複面積
            overlap = np.sum(np.minimum(hist1, hist2)) / len(bins)
            
            return overlap
            
        except Exception:
            return 0.5  # デフォルト値
    
    def generate_anomaly_report(self, data_df, output_path=None):
        """
        異常検知レポート生成
        
        Args:
            data_df (pd.DataFrame): 元データ
            output_path (str): 出力パス
            
        Returns:
            dict: レポートデータ
        """
        if not self.is_fitted:
            self.logger.error("❌ 異常検知が未実行です")
            return {}
        
        self.logger.info("📝 異常検知レポート生成開始")
        
        try:
            # 異常グリッド詳細分析
            anomaly_details = self._analyze_anomaly_details(data_df)
            
            # 地理的分布分析
            geographic_analysis = self._analyze_geographic_distribution(data_df)
            
            # 時系列パターン分析（月別データがある場合）
            temporal_analysis = self._analyze_temporal_patterns(data_df)
            
            # レポート統合
            report = {
                'detection_summary': self.detection_results['summary'],
                'anomaly_details': anomaly_details,
                'geographic_analysis': geographic_analysis,
                'temporal_analysis': temporal_analysis,
                'feature_importance': self.feature_importance,
                'model_performance': self._evaluate_model_performance(),
                'recommendations': self._generate_recommendations(),
                'metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'total_features': len(self.feature_names) if self.feature_names else 0,
                    'model_parameters': self.config
                }
            }
            
            # ファイル出力
            if output_path:
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                self.logger.info(f"✅ レポート保存: {output_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ レポート生成エラー: {e}")
            return {}
    
    def _analyze_anomaly_details(self, data_df):
        """異常グリッド詳細分析"""
        predictions = self.detection_results['predictions']
        anomaly_scores = self.detection_results['anomaly_scores']
        
        # 異常グリッドの抽出
        anomaly_mask = predictions == -1
        anomaly_data = data_df[anomaly_mask].copy()
        anomaly_data['anomaly_score'] = anomaly_scores[anomaly_mask]
        
        if len(anomaly_data) == 0:
            return {'message': '異常グリッドが検出されませんでした'}
        
        # 異常強度による分類
        score_threshold_critical = np.percentile(anomaly_scores[anomaly_mask], 10)  # 下位10%
        score_threshold_high = np.percentile(anomaly_scores[anomaly_mask], 30)     # 下位30%
        
        critical_anomalies = anomaly_data[anomaly_data['anomaly_score'] <= score_threshold_critical]
        high_anomalies = anomaly_data[
            (anomaly_data['anomaly_score'] > score_threshold_critical) & 
            (anomaly_data['anomaly_score'] <= score_threshold_high)
        ]
        moderate_anomalies = anomaly_data[anomaly_data['anomaly_score'] > score_threshold_high]
        
        details = {
            'total_anomalies': len(anomaly_data),
            'severity_distribution': {
                'critical': len(critical_anomalies),
                'high': len(high_anomalies),
                'moderate': len(moderate_anomalies)
            },
            'score_statistics': {
                'min_score': float(np.min(anomaly_scores[anomaly_mask])),
                'max_score': float(np.max(anomaly_scores[anomaly_mask])),
                'mean_score': float(np.mean(anomaly_scores[anomaly_mask])),
                'std_score': float(np.std(anomaly_scores[anomaly_mask]))
            },
            'top_anomalies': self._get_top_anomalies(anomaly_data, top_n=10)
        }
        
        return details
    
    def _get_top_anomalies(self, anomaly_data, top_n=10):
        """上位異常グリッドの取得"""
        top_anomalies = anomaly_data.nsmallest(top_n, 'anomaly_score')
        
        results = []
        for idx, row in top_anomalies.iterrows():
            anomaly_info = {
                'grid_id': idx,
                'anomaly_score': float(row['anomaly_score']),
                'location': {
                    'latitude': float(row.get('latitude', 0)),
                    'longitude': float(row.get('longitude', 0)),
                    'continent': row.get('continent', 'Unknown')
                },
                'key_features': {}
            }
            
            # 重要特徴量の値
            if self.feature_importance:
                top_features = list(self.feature_importance.keys())[:5]
                for feature in top_features:
                    if feature in row:
                        anomaly_info['key_features'][feature] = float(row[feature])
            
            results.append(anomaly_info)
        
        return results
    
    def _analyze_geographic_distribution(self, data_df):
        """地理的分布分析"""
        predictions = self.detection_results['predictions']
        
        try:
            # 大陸別分析
            if 'continent' in data_df.columns:
                continent_analysis = data_df.groupby('continent').agg({
                    'latitude': 'count',
                }).rename(columns={'latitude': 'total_grids'})
                
                # 異常グリッド数
                anomaly_mask = predictions == -1
                anomaly_by_continent = data_df[anomaly_mask].groupby('continent').size()
                continent_analysis['anomaly_grids'] = anomaly_by_continent.fillna(0)
                continent_analysis['anomaly_rate'] = continent_analysis['anomaly_grids'] / continent_analysis['total_grids']
                
                geographic_analysis = {
                    'continent_distribution': continent_analysis.to_dict('index'),
                    'highest_risk_continent': continent_analysis['anomaly_rate'].idxmax(),
                    'lowest_risk_continent': continent_analysis['anomaly_rate'].idxmin()
                }
            else:
                geographic_analysis = {'message': '大陸データが利用できません'}
            
            return geographic_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_temporal_patterns(self, data_df):
        """時系列パターン分析"""
        try:
            predictions = self.detection_results['predictions']
            anomaly_mask = predictions == -1
            
            # 月別火災データがある場合の分析
            month_columns = [col for col in data_df.columns if col.startswith('month_') and col.endswith('_fire')]
            
            if month_columns:
                # 月別異常パターン
                anomaly_data = data_df[anomaly_mask]
                normal_data = data_df[~anomaly_mask]
                
                monthly_patterns = {}
                for month_col in month_columns:
                    month_num = int(month_col.split('_')[1])
                    
                    anomaly_mean = anomaly_data[month_col].mean() if len(anomaly_data) > 0 else 0
                    normal_mean = normal_data[month_col].mean() if len(normal_data) > 0 else 0
                    
                    monthly_patterns[month_num] = {
                        'anomaly_mean': float(anomaly_mean),
                        'normal_mean': float(normal_mean),
                        'difference': float(anomaly_mean - normal_mean)
                    }
                
                # 最も異常が顕著な月
                max_diff_month = max(monthly_patterns.items(), key=lambda x: abs(x[1]['difference']))
                
                temporal_analysis = {
                    'monthly_patterns': monthly_patterns,
                    'peak_anomaly_month': max_diff_month[0],
                    'seasonal_effect': 'detected' if len(monthly_patterns) > 6 else 'insufficient_data'
                }
            else:
                temporal_analysis = {'message': '時系列データが利用できません'}
            
            return temporal_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _evaluate_model_performance(self):
        """モデル性能評価"""
        if not self.detection_results:
            return {}
        
        try:
            # 分離品質メトリクス
            separation_metrics = self.detection_results.get('separation_metrics', {})
            
            # 検知率評価
            target_anomaly_rate = self.config['contamination']
            actual_anomaly_rate = self.detection_results['summary']['anomaly_rate']
            rate_accuracy = 1 - abs(target_anomaly_rate - actual_anomaly_rate) / target_anomaly_rate
            
            # 総合品質スコア
            quality_components = {
                'rate_accuracy': rate_accuracy,
                'separation_quality': min(1.0, separation_metrics.get('effect_size', 0) / 2),
                'feature_utilization': len(self.feature_importance) / len(self.feature_names) if self.feature_names else 0
            }
            
            overall_quality = np.mean(list(quality_components.values()))
            
            performance = {
                'quality_components': quality_components,
                'overall_quality_score': float(overall_quality),
                'separation_metrics': separation_metrics,
                'detection_efficiency': {
                    'target_anomaly_rate': target_anomaly_rate,
                    'actual_anomaly_rate': float(actual_anomaly_rate),
                    'rate_accuracy': float(rate_accuracy)
                }
            }
            
            return performance
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self):
        """推奨事項生成"""
        recommendations = {
            'model_optimization': [],
            'monitoring_enhancement': [],
            'data_quality': []
        }
        
        # モデル最適化推奨
        if self.detection_results:
            anomaly_rate = self.detection_results['summary']['anomaly_rate']
            target_rate = self.config['contamination']
            
            if abs(anomaly_rate - target_rate) / target_rate > 0.3:
                recommendations['model_optimization'].append(
                    f"異常検知率調整が必要 (現在: {anomaly_rate:.1%}, 目標: {target_rate:.1%})"
                )
            
            if len(self.feature_importance) < 5:
                recommendations['model_optimization'].append(
                    "特徴量の追加または特徴工学の改善を検討"
                )
        
        # 監視強化推奨
        recommendations['monitoring_enhancement'].extend([
            "異常スコア閾値の動的調整システム導入",
            "リアルタイム異常アラート機能の実装",
            "異常パターンの自動分類システム構築"
        ])
        
        # データ品質推奨
        recommendations['data_quality'].extend([
            "特徴量の相関分析と冗長性除去",
            "外れ値の事前処理手法改善",
            "データ収集頻度の最適化"
        ])
        
        return recommendations
    
    def visualize_anomaly_detection(self, data_df, save_path=None):
        """異常検知結果の可視化"""
        if not self.is_fitted:
            self.logger.error("❌ 異常検知が未実行です")
            return
        
        self.logger.info("📊 異常検知可視化開始")
        
        try:
            # 日本語フォント設定
            plt.rcParams['font.family'] = ['Yu Gothic', 'Hiragino Kaku Gothic Pro', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            predictions = self.detection_results['predictions']
            anomaly_scores = self.detection_results['anomaly_scores']
            
            # 4パネル可視化
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Isolation Forest Fire Anomaly Detection Results', fontsize=16, fontweight='bold')
            
            # 1. 異常スコア分布
            axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].axvline(np.percentile(anomaly_scores, 10), color='red', linestyle='--', 
                              label=f'Anomaly Threshold (10%)')
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Anomaly Score Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 地理的分布
            if 'latitude' in data_df.columns and 'longitude' in data_df.columns:
                normal_mask = predictions == 1
                anomaly_mask = predictions == -1
                
                axes[0, 1].scatter(data_df.loc[normal_mask, 'longitude'], 
                                  data_df.loc[normal_mask, 'latitude'],
                                  c='lightblue', alpha=0.5, s=10, label='Normal')
                axes[0, 1].scatter(data_df.loc[anomaly_mask, 'longitude'], 
                                  data_df.loc[anomaly_mask, 'latitude'],
                                  c='red', alpha=0.8, s=30, label='Anomaly')
                axes[0, 1].set_xlabel('Longitude')
                axes[0, 1].set_ylabel('Latitude')
                axes[0, 1].set_title('Geographic Distribution')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 特徴量重要度
            if self.feature_importance:
                top_features = list(self.feature_importance.keys())[:8]
                importance_scores = [self.feature_importance[f]['importance_score'] for f in top_features]
                
                axes[1, 0].barh(range(len(top_features)), importance_scores, color='green', alpha=0.7)
                axes[1, 0].set_yticks(range(len(top_features)))
                axes[1, 0].set_yticklabels([f.replace('_', ' ').title() for f in top_features])
                axes[1, 0].set_xlabel('Importance Score')
                axes[1, 0].set_title('Feature Importance for Anomaly Detection')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 大陸別異常率
            if 'continent' in data_df.columns:
                continent_stats = data_df.groupby('continent').agg({'latitude': 'count'}).rename(columns={'latitude': 'total'})
                anomaly_by_continent = data_df[predictions == -1].groupby('continent').size()
                continent_stats['anomalies'] = anomaly_by_continent.fillna(0)
                continent_stats['anomaly_rate'] = continent_stats['anomalies'] / continent_stats['total']
                
                axes[1, 1].bar(range(len(continent_stats)), continent_stats['anomaly_rate'], 
                              color='orange', alpha=0.7)
                axes[1, 1].set_xticks(range(len(continent_stats)))
                axes[1, 1].set_xticklabels(continent_stats.index, rotation=45, ha='right')
                axes[1, 1].set_ylabel('Anomaly Rate')
                axes[1, 1].set_title('Anomaly Rate by Continent')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ 可視化保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"❌ 可視化エラー: {e}")


if __name__ == "__main__":
    # テスト実行
    print("🔍 Fire Anomaly Detector テスト")
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'latitude': np.random.uniform(-60, 70, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'continent': np.random.choice(['Africa', 'Asia', 'North America', 'Europe'], n_samples),
        'fire_activity': np.random.gamma(2, 2, n_samples),
        'burned_area_total': np.random.exponential(5, n_samples),
        'neighbor_max': np.random.gamma(3, 3, n_samples),
        'neighbor_std': np.random.gamma(2, 1, n_samples),
        'temperature_avg': np.random.normal(25, 10, n_samples),
        'precipitation_total': np.random.exponential(50, n_samples)
    })
    
    # 異常検知器初期化
    detector = FireAnomalyDetector()
    
    # 特徴量準備
    features = detector.prepare_features(sample_data)
    
    # 異常検知実行
    results = detector.detect_anomalies(features)
    
    # レポート生成
    sample_data['anomaly_prediction'] = results['predictions']
    sample_data['anomaly_score'] = results['anomaly_scores']
    
    report = detector.generate_anomaly_report(sample_data)
    
    print(f"✅ 異常検知完了: {results['summary']['anomalous_count']}個検出")
    print(f"📊 異常率: {results['summary']['anomaly_rate']:.1%}")