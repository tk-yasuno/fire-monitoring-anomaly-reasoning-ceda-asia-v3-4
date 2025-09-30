#!/usr/bin/env python3
"""
Asia-Pacific Fire Monitoring and Anomaly Reasoning System v3.4
NASA FIRMS + CEDA Fire_cci 邨ｱ蜷育屮隕悶す繧ｹ繝・Β + 逡ｰ蟶ｸ讀懃衍 + MiniCPM謗ｨ隲・

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

# 讖滓｢ｰ蟄ｦ鄙偵Λ繧､繝悶Λ繝ｪ
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 繧ｷ繧ｹ繝・Β繝代せ險ｭ螳・
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

# v3.2繧ｷ繧ｹ繝・Β縺九ｉ縺ｮ邯呎価逕ｨ
try:
    # v3.2縺ｮ繧ｳ繧｢繧ｷ繧ｹ繝・Β繧貞茜逕ｨ
    sys.path.append(str(Path(__file__).parent.parent / 'global-fire-monitoring-v3-0'))
    from global_fire_monitoring_v32 import GlobalFireMonitoringSystemV32
    V32_AVAILABLE = True
except ImportError:
    V32_AVAILABLE = False
    print("笞・・v3.2繧ｷ繧ｹ繝・Β縺悟茜逕ｨ縺ｧ縺阪∪縺帙ｓ")

class GlobalFireMonitoringAndAnomalyReasoningSystemV33:
    """
    Asia-Pacific Fire Monitoring and Anomaly Reasoning System v3.4
    
    v3.2縺ｮ讖溯・縺ｫ蜉縺医※莉･荳九ｒ霑ｽ蜉・・
    - Isolation Forest 縺ｫ繧医ｋ逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ讀懃衍
    - MiniCPM 縺ｫ繧医ｋ逡ｰ蟶ｸ逅・罰謗ｨ隲・
    - 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ蜿ｯ隕門喧縺ｨ蛻・梵
    """
    
    def __init__(self, config_path='config/global_config_v33.json'):
        """
        蛻晄悄蛹・
        
        Args:
            config_path (str): 險ｭ螳壹ヵ繧｡繧､繝ｫ繝代せ
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # v3.2繧ｷ繧ｹ繝・Β縺ｮ蛻晄悄蛹・
        if V32_AVAILABLE:
            try:
                self.v32_system = GlobalFireMonitoringSystemV32()
                self.logger.info("笨・v3.2繧ｷ繧ｹ繝・Β蛻晄悄蛹匁・蜉・)
            except Exception as e:
                self.logger.warning(f"笞・・v3.2繧ｷ繧ｹ繝・Β蛻晄悄蛹門､ｱ謨・ {e}")
                self.v32_system = None
        else:
            self.v32_system = None
            
        # 逡ｰ蟶ｸ讀懃衍繧ｳ繝ｳ繝昴・繝阪Φ繝・
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # MiniCPM謗ｨ隲悶お繝ｳ繧ｸ繝ｳ・医・繝ｬ繝ｼ繧ｹ繝帙Ν繝繝ｼ・・
        self.reasoning_engine = None
        
        # 邨先棡菫晏ｭ倡畑
        self.latest_results = {}
        
        self.logger.info("櫨 Asia-Pacific Fire Monitoring and Anomaly Reasoning System v3.4 蛻晄悄蛹門ｮ御ｺ・)
    
    def _setup_logging(self):
        """繝ｭ繧ｰ險ｭ螳・""
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
        """險ｭ螳壹ヵ繧｡繧､繝ｫ隱ｭ縺ｿ霎ｼ縺ｿ"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"笨・險ｭ螳壹ヵ繧｡繧､繝ｫ隱ｭ縺ｿ霎ｼ縺ｿ謌仙粥: {self.config_path}")
                return config
            else:
                self.logger.warning(f"笞・・險ｭ螳壹ヵ繧｡繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {self.config_path}")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"笶・險ｭ螳壹ヵ繧｡繧､繝ｫ隱ｭ縺ｿ霎ｼ縺ｿ繧ｨ繝ｩ繝ｼ: {e}")
            return self._create_default_config()
    
    def _create_default_config(self):
        """繝・ヵ繧ｩ繝ｫ繝郁ｨｭ螳壹・菴懈・"""
        default_config = {
            "system": {
                "name": "Asia-Pacific Fire Monitoring and Anomaly Reasoning System",
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
        
        # 險ｭ螳壹ヵ繧｡繧､繝ｫ繧剃ｿ晏ｭ・
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"笨・繝・ヵ繧ｩ繝ｫ繝郁ｨｭ螳壹ヵ繧｡繧､繝ｫ菴懈・: {self.config_path}")
        return default_config
    
    def process_yearly_data_with_anomaly_detection(self, year=2022):
        """
        蟷ｴ谺｡繝・・繧ｿ蜃ｦ逅・+ 逡ｰ蟶ｸ讀懃衍
        
        Args:
            year (int): 蜃ｦ逅・ｹｴ
            
        Returns:
            dict: 蜃ｦ逅・ｵ先棡・磯壼ｸｸ繝・・繧ｿ + 逡ｰ蟶ｸ讀懃衍邨先棡・・
        """
        self.logger.info(f"剥 {year}蟷ｴ繝・・繧ｿ縺ｮ逡ｰ蟶ｸ讀懃衍蜃ｦ逅・幕蟋・)
        
        # v3.2繧ｷ繧ｹ繝・Β縺ｧ繝吶・繧ｹ繝・・繧ｿ蜃ｦ逅・
        if self.v32_system:
            try:
                base_results = self.v32_system.process_yearly_data(year=year)
                self.logger.info("笨・v3.2繝吶・繧ｹ繝・・繧ｿ蜃ｦ逅・ｮ御ｺ・)
            except Exception as e:
                self.logger.error(f"笶・v3.2繝・・繧ｿ蜃ｦ逅・お繝ｩ繝ｼ: {e}")
                # 繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ・壹す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・・繧ｿ
                base_results = self._generate_simulation_data()
        else:
            # v3.2縺悟茜逕ｨ縺ｧ縺阪↑縺・ｴ蜷医・繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ
            base_results = self._generate_simulation_data()
        
        # 逡ｰ蟶ｸ讀懃衍縺ｮ螳溯｡・
        anomaly_results = self._detect_anomalies(base_results)
        
        # 邨先棡縺ｮ邨ｱ蜷・
        combined_results = {
            'base_data': base_results,
            'anomaly_detection': anomaly_results,
            'processing_timestamp': datetime.now().isoformat(),
            'year': year
        }
        
        self.latest_results = combined_results
        return combined_results
    
    def _generate_simulation_data(self):
        """繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・・繧ｿ逕滓・・・3.2縺悟茜逕ｨ縺ｧ縺阪↑縺・ｴ蜷茨ｼ・""
        self.logger.info("売 繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・・繧ｿ逕滓・荳ｭ...")
        
        # 12,500繧ｰ繝ｪ繝・ラ縺ｮ繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・・繧ｿ
        n_samples = 12500
        
        # 螟ｧ髯ｸ縺斐→縺ｮ繧ｵ繝ｳ繝励Ν謨ｰ
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
            # 螟ｧ髯ｸ縺斐→縺ｮ迚ｹ蠕ｴ繝代ち繝ｼ繝ｳ
            if continent == 'Africa':
                # 繧｢繝輔Μ繧ｫ・夐ｫ倥＞轣ｫ轣ｽ豢ｻ蜍・
                base_fire_activity = np.random.gamma(3, 2, n_cont_samples)
                lat_range = (-35, 37)
                lon_range = (-20, 55)
            elif continent == 'North America':
                # 蛹礼ｱｳ・壻ｸｭ遞句ｺｦ縲∝ｭ｣遽螟牙虚螟ｧ
                base_fire_activity = np.random.beta(2, 5, n_cont_samples) * 10
                lat_range = (25, 75)
                lon_range = (-170, -50)
            elif continent == 'Asia':
                # 繧｢繧ｸ繧｢・壼､壽ｧ倥↑繝代ち繝ｼ繝ｳ
                base_fire_activity = np.random.lognormal(1, 1.5, n_cont_samples)
                lat_range = (-10, 70)
                lon_range = (60, 180)
            else:
                # 縺昴・莉悶・螟ｧ髯ｸ
                base_fire_activity = np.random.exponential(2, n_cont_samples)
                lat_range = (-50, 70)
                lon_range = (-180, 180)
            
            # 蠎ｧ讓咏函謌・
            lats = np.random.uniform(lat_range[0], lat_range[1], n_cont_samples)
            lons = np.random.uniform(lon_range[0], lon_range[1], n_cont_samples)
            
            # 24谺｡蜈・音蠕ｴ驥冗函謌・
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
                
                # 霑大ｍ迚ｹ蠕ｴ驥擾ｼ磯㍾隕∝ｺｦ縺碁ｫ倥＞・・
                features.update({
                    'neighbor_max': features['fire_activity'] * np.random.uniform(1.2, 3.0),
                    'neighbor_std': features['fire_activity'] * np.random.uniform(0.2, 0.8),
                    'neighbor_mean': features['fire_activity'] * np.random.uniform(0.8, 1.5),
                    'neighbor_count': np.random.poisson(5),
                })
                
                # 譛亥挨繝代ち繝ｼ繝ｳ・・2繝ｶ譛茨ｼ・
                seasonal_pattern = np.sin(np.arange(12) * 2 * np.pi / 12) + 1
                for month in range(12):
                    features[f'month_{month+1}_fire'] = (
                        features['fire_activity'] * seasonal_pattern[month] * 
                        np.random.uniform(0.5, 1.5)
                    )
                
                data_list.append(features)
        
        df = pd.DataFrame(data_list)
        
        self.logger.info(f"笨・繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・・繧ｿ逕滓・螳御ｺ・ {len(df)}繧ｰ繝ｪ繝・ラ")
        return {'processed_data': df, 'metadata': {'data_type': 'simulation'}}
    
    def _detect_anomalies(self, base_results):
        """
        Isolation Forest縺ｫ繧医ｋ逡ｰ蟶ｸ讀懃衍
        
        Args:
            base_results (dict): v3.2縺九ｉ縺ｮ蝓ｺ譛ｬ蜃ｦ逅・ｵ先棡
            
        Returns:
            dict: 逡ｰ蟶ｸ讀懃衍邨先棡
        """
        self.logger.info("剥 Isolation Forest逡ｰ蟶ｸ讀懃衍髢句ｧ・)
        
        try:
            # 繝・・繧ｿ縺ｮ貅門ｙ
            if 'processed_data' in base_results:
                df = base_results['processed_data']
            else:
                raise ValueError("蜃ｦ逅・ｸ医∩繝・・繧ｿ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ")
            
            # 迚ｹ蠕ｴ驥上・驕ｸ謚橸ｼ域焚蛟､繝・・繧ｿ縺ｮ縺ｿ・・
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            feature_data = df[numeric_columns].copy()
            
            # 谺謳榊､縺ｮ蜃ｦ逅・
            feature_data = feature_data.fillna(feature_data.median())
            
            self.logger.info(f"投 迚ｹ蠕ｴ驥乗焚: {len(feature_data.columns)}")
            self.logger.info(f"投 繧ｵ繝ｳ繝励Ν謨ｰ: {len(feature_data)}")
            
            # 繝・・繧ｿ縺ｮ讓呎ｺ門喧
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Isolation Forest險ｭ螳・
            anomaly_config = self.config['anomaly_detection']
            
            self.anomaly_detector = IsolationForest(
                contamination=anomaly_config['contamination'],
                n_estimators=anomaly_config['n_estimators'],
                random_state=anomaly_config['random_state'],
                bootstrap=anomaly_config['bootstrap'],
                n_jobs=-1
            )
            
            # 逡ｰ蟶ｸ讀懃衍螳溯｡・
            predictions = self.anomaly_detector.fit_predict(scaled_features)
            anomaly_scores = self.anomaly_detector.score_samples(scaled_features)
            
            # 邨先棡縺ｮ邨ｱ蜷・
            df['anomaly_prediction'] = predictions  # -1: 逡ｰ蟶ｸ, 1: 豁｣蟶ｸ
            df['anomaly_score'] = anomaly_scores    # 繧ｹ繧ｳ繧｢・井ｽ弱＞縺ｻ縺ｩ逡ｰ蟶ｸ・・
            df['is_anomaly'] = (predictions == -1)
            
            # 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ邨ｱ險・
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions)
            
            # 螟ｧ髯ｸ蛻･逡ｰ蟶ｸ蛻・ｸ・
            continent_anomalies = df.groupby('continent')['is_anomaly'].agg(['sum', 'count', 'mean'])
            
            # 迚ｹ蠕ｴ驥城㍾隕∝ｺｦ・育焚蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ迚ｹ蠕ｴ蛻・梵・・
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
            
            # 驥崎ｦ∝ｺｦ縺ｧ繧ｽ繝ｼ繝・
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ縺ｿ繧呈歓蜃ｺ
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
                'feature_importance': dict(sorted_features[:10]),  # 荳贋ｽ・0迚ｹ蠕ｴ
                'anomaly_data': df,  # 蜈ｨ繝・・繧ｿ・育焚蟶ｸ繝輔Λ繧ｰ莉倥″・・
                'anomaly_grids': anomaly_grids,  # 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ縺ｿ
                'model_parameters': anomaly_config
            }
            
            self.logger.info(f"笨・逡ｰ蟶ｸ讀懃衍螳御ｺ・ {n_anomalies}蛟九・逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ讀懷・ ({anomaly_rate:.1%})")
            return anomaly_results
            
        except Exception as e:
            self.logger.error(f"笶・逡ｰ蟶ｸ讀懃衍繧ｨ繝ｩ繝ｼ: {e}")
            return {'error': str(e)}
    
    def analyze_anomalies_with_reasoning(self, top_n=10):
        """
        逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ隧ｳ邏ｰ蛻・梵縺ｨMiniCPM謗ｨ隲・
        
        Args:
            top_n (int): 蛻・梵縺吶ｋ荳贋ｽ咲焚蟶ｸ繧ｰ繝ｪ繝・ラ謨ｰ
            
        Returns:
            dict: 謗ｨ隲也ｵ先棡
        """
        if not self.latest_results or 'anomaly_detection' not in self.latest_results:
            self.logger.error("笶・逡ｰ蟶ｸ讀懃衍邨先棡縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ")
            return {}
        
        self.logger.info(f"､・荳贋ｽ砿top_n}逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ謗ｨ隲門・譫宣幕蟋・)
        
        try:
            anomaly_data = self.latest_results['anomaly_detection']['anomaly_data']
            anomaly_grids = anomaly_data[anomaly_data['is_anomaly']].copy()
            
            # 逡ｰ蟶ｸ繧ｹ繧ｳ繧｢縺ｧ荳ｦ縺ｹ譖ｿ縺茨ｼ井ｽ弱＞縺ｻ縺ｩ逡ｰ蟶ｸ・・
            anomaly_grids = anomaly_grids.sort_values('anomaly_score').head(top_n)
            
            reasoning_results = []
            
            for idx, row in anomaly_grids.iterrows():
                # 蜷・焚蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ蛻・梵
                analysis = self._analyze_single_anomaly(row)
                
                # MiniCPM謗ｨ隲厄ｼ医・繝ｬ繝ｼ繧ｹ繝帙Ν繝繝ｼ螳溯｣・ｼ・
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
            
            self.logger.info(f"笨・{len(reasoning_results)}蛟九・逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ謗ｨ隲門ｮ御ｺ・)
            
            return {
                'reasoning_results': reasoning_results,
                'summary': self._summarize_anomaly_patterns(reasoning_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"笶・逡ｰ蟶ｸ謗ｨ隲悶お繝ｩ繝ｼ: {e}")
            return {'error': str(e)}
    
    def _analyze_single_anomaly(self, grid_data):
        """蜊倅ｸ逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ蛻・梵"""
        analysis = {
            'key_features': {},
            'deviation_factors': [],
            'risk_level': 'unknown'
        }
        
        # 驥崎ｦ∫音蠕ｴ驥上・蛻・梵
        important_features = ['fire_activity', 'burned_area_total', 'neighbor_max', 'neighbor_std']
        
        for feature in important_features:
            if feature in grid_data:
                value = grid_data[feature]
                analysis['key_features'][feature] = value
                
                # 蛛丞ｷｮ隕∝屏縺ｮ蛻､螳・
                if feature == 'fire_activity' and value > 10:
                    analysis['deviation_factors'].append('讌ｵ遶ｯ縺ｫ鬮倥＞轣ｫ轣ｽ豢ｻ蜍・)
                elif feature == 'neighbor_max' and value > 15:
                    analysis['deviation_factors'].append('霑大ｍ縺ｧ逡ｰ蟶ｸ縺ｪ鬮倡↓轣ｽ蛟､')
                elif feature == 'neighbor_std' and value > 5:
                    analysis['deviation_factors'].append('霑大ｍ轣ｫ轣ｽ縺ｮ鬮倥＞螟牙虚諤ｧ')
        
        # 繝ｪ繧ｹ繧ｯ繝ｬ繝吶Ν蛻､螳・
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
        MiniCPM謗ｨ隲悶・逕滓・・医・繝ｬ繝ｼ繧ｹ繝帙Ν繝繝ｼ螳溯｣・ｼ・
        
        螳滄圀縺ｮ螳溯｣・〒縺ｯ縲｀iniCPM繝｢繝・Ν繧剃ｽｿ逕ｨ縺励※謗ｨ隲悶ｒ逕滓・
        """
        # 繝励Ξ繝ｼ繧ｹ繝帙Ν繝繝ｼ謗ｨ隲悶Ο繧ｸ繝・け
        continent = grid_data.get('continent', 'Unknown')
        fire_activity = grid_data.get('fire_activity', 0)
        risk_level = analysis['risk_level']
        deviation_factors = analysis['deviation_factors']
        
        # 蝓ｺ譛ｬ逧・↑謗ｨ隲悶ユ繝ｳ繝励Ξ繝ｼ繝・
        reasoning_templates = {
            'critical': f"{continent}縺ｮ隧ｲ蠖薙げ繝ｪ繝・ラ縺ｯ讌ｵ繧√※豺ｱ蛻ｻ縺ｪ轣ｫ轣ｽ逡ｰ蟶ｸ繧堤､ｺ縺励※縺・ｋ縲・,
            'high': f"{continent}縺ｫ縺翫＞縺ｦ鬮倥＞轣ｫ轣ｽ繝ｪ繧ｹ繧ｯ縺梧､懷・縺輔ｌ縺溘・,
            'moderate': f"{continent}縺ｧ荳ｭ遞句ｺｦ縺ｮ轣ｫ轣ｽ逡ｰ蟶ｸ繝代ち繝ｼ繝ｳ縺瑚ｦｳ貂ｬ縺輔ｌ縺溘・,
            'low': f"{continent}縺ｧ霆ｽ蠕ｮ縺ｪ轣ｫ轣ｽ繝代ち繝ｼ繝ｳ縺ｮ逡ｰ蟶ｸ縺檎｢ｺ隱阪＆繧後◆縲・
        }
        
        base_reasoning = reasoning_templates.get(risk_level, "逡ｰ蟶ｸ繝代ち繝ｼ繝ｳ縺梧､懷・縺輔ｌ縺溘・)
        
        # 蛛丞ｷｮ隕∝屏縺ｮ霑ｽ蜉
        if deviation_factors:
            factors_text = "縲・.join(deviation_factors)
            detailed_reasoning = f"{base_reasoning} 荳ｻ縺ｪ隕∝屏: {factors_text}縲・
        else:
            detailed_reasoning = f"{base_reasoning} 隧ｳ邏ｰ縺ｪ隕∝屏蛻・梵縺悟ｿ・ｦ√・
        
        # 謗ｨ螂ｨ繧｢繧ｯ繧ｷ繝ｧ繝ｳ
        if risk_level == 'critical':
            action = "蜊ｳ蠎ｧ縺ｮ迴ｾ蝨ｰ隱ｿ譟ｻ縺ｨ邱頑･蟇ｾ蠢應ｽ灘宛縺ｮ遒ｺ遶九ｒ謗ｨ螂ｨ縲・
        elif risk_level == 'high':
            action = "譌ｩ譛溘・迴ｾ蝨ｰ遒ｺ隱阪→逶｣隕門ｼｷ蛹悶ｒ謗ｨ螂ｨ縲・
        else:
            action = "邯咏ｶ夂噪縺ｪ逶｣隕悶→螳壽悄逧・↑繝輔か繝ｭ繝ｼ繧｢繝・・繧呈耳螂ｨ縲・
        
        full_reasoning = f"{detailed_reasoning} {action}"
        
        return {
            'explanation': full_reasoning,
            'confidence': 0.75,  # 繝励Ξ繝ｼ繧ｹ繝帙Ν繝繝ｼ菫｡鬆ｼ蠎ｦ
            'model': 'minicpm_placeholder',
            'factors_identified': deviation_factors,
            'recommended_action': action
        }
    
    def _summarize_anomaly_patterns(self, reasoning_results):
        """逡ｰ蟶ｸ繝代ち繝ｼ繝ｳ縺ｮ隕∫ｴ・""
        if not reasoning_results:
            return {}
        
        # 繝ｪ繧ｹ繧ｯ繝ｬ繝吶Ν蛻・ｸ・
        risk_levels = [r['analysis']['risk_level'] for r in reasoning_results]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        # 螟ｧ髯ｸ蛻・ｸ・
        continents = [r['location']['continent'] for r in reasoning_results]
        continent_distribution = {cont: continents.count(cont) for cont in set(continents)}
        
        # 蜈ｱ騾夊ｦ∝屏
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
        """逡ｰ蟶ｸ讀懃衍邨先棡縺ｮ蜿ｯ隕門喧"""
        if not self.latest_results or 'anomaly_detection' not in self.latest_results:
            self.logger.error("笶・蜿ｯ隕門喧逕ｨ繝・・繧ｿ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ")
            return
        
        self.logger.info("投 逡ｰ蟶ｸ讀懃衍蜿ｯ隕門喧菴懈・髢句ｧ・)
        
        try:
            # 譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝郁ｨｭ螳・
            plt.rcParams['font.family'] = ['Yu Gothic', 'Hiragino Kaku Gothic Pro', 'Takao Gothic', 'Droid Sans Fallback', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 繝・・繧ｿ縺ｮ貅門ｙ
            anomaly_data = self.latest_results['anomaly_detection']['anomaly_data']
            
            # 6繝代ロ繝ｫ蜿ｯ隕門喧
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Asia-Pacific Fire Monitoring and Anomaly Detection Analysis v3.4', fontsize=16, fontweight='bold')
            
            # 1. 逡ｰ蟶ｸ繧ｹ繧ｳ繧｢蛻・ｸ・
            axes[0, 0].hist(anomaly_data['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(anomaly_data['anomaly_score'].quantile(0.1), color='red', linestyle='--', label='Anomaly Threshold')
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Anomaly Score Distribution')
            axes[0, 0].legend()
            
            # 2. 蝨ｰ逅・噪蛻・ｸ・ｼ育焚蟶ｸ繧ｰ繝ｪ繝・ラ・・
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
            
            # 3. 螟ｧ髯ｸ蛻･逡ｰ蟶ｸ邇・
            continent_stats = anomaly_data.groupby('continent')['is_anomaly'].agg(['sum', 'count'])
            continent_stats['anomaly_rate'] = continent_stats['sum'] / continent_stats['count']
            
            axes[0, 2].bar(range(len(continent_stats)), continent_stats['anomaly_rate'], 
                          color='orange', alpha=0.7)
            axes[0, 2].set_xticks(range(len(continent_stats)))
            axes[0, 2].set_xticklabels(continent_stats.index, rotation=45, ha='right')
            axes[0, 2].set_ylabel('Anomaly Rate')
            axes[0, 2].set_title('Anomaly Rate by Continent')
            
            # 4. 轣ｫ轣ｽ豢ｻ蜍・vs 逡ｰ蟶ｸ繧ｹ繧ｳ繧｢
            axes[1, 0].scatter(anomaly_data['fire_activity'], anomaly_data['anomaly_score'], 
                             c=anomaly_data['is_anomaly'], cmap='RdYlBu', alpha=0.6)
            axes[1, 0].set_xlabel('Fire Activity')
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Fire Activity vs Anomaly Score')
            
            # 5. 迚ｹ蠕ｴ驥城㍾隕∝ｺｦ
            if 'feature_importance' in self.latest_results['anomaly_detection']:
                importance = self.latest_results['anomaly_detection']['feature_importance']
                features = list(importance.keys())[:8]  # 荳贋ｽ・迚ｹ蠕ｴ
                values = [importance[f] for f in features]
                
                axes[1, 1].barh(range(len(features)), values, color='green', alpha=0.7)
                axes[1, 1].set_yticks(range(len(features)))
                axes[1, 1].set_yticklabels(features)
                axes[1, 1].set_xlabel('Importance Score')
                axes[1, 1].set_title('Feature Importance for Anomaly Detection')
            
            # 6. 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ隧ｳ邏ｰ邨ｱ險・
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
            
            # 菫晏ｭ・
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/anomaly_detection_analysis_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"笨・蜿ｯ隕門喧菫晏ｭ伜ｮ御ｺ・ {output_path}")
            
        except Exception as e:
            self.logger.error(f"笶・蜿ｯ隕門喧繧ｨ繝ｩ繝ｼ: {e}")
    
    def run_complete_anomaly_analysis(self, year=2022, top_anomalies=10):
        """
        螳悟・縺ｪ逡ｰ蟶ｸ讀懃衍蛻・梵繝代う繝励Λ繧､繝ｳ
        
        Args:
            year (int): 蛻・梵蟷ｴ
            top_anomalies (int): 隧ｳ邏ｰ蛻・梵縺吶ｋ逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ謨ｰ
            
        Returns:
            dict: 螳悟・縺ｪ蛻・梵邨先棡
        """
        self.logger.info(f"噫 Asia-Pacific Fire Monitoring and Anomaly Analysis v3.4 螳悟・螳溯｡碁幕蟋・)
        
        try:
            # 1. 繝・・繧ｿ蜃ｦ逅・+ 逡ｰ蟶ｸ讀懃衍
            results = self.process_yearly_data_with_anomaly_detection(year=year)
            
            # 2. 逡ｰ蟶ｸ繧ｰ繝ｪ繝・ラ縺ｮ謗ｨ隲門・譫・
            reasoning_results = self.analyze_anomalies_with_reasoning(top_n=top_anomalies)
            
            # 3. 蜿ｯ隕門喧
            self.create_anomaly_visualization()
            
            # 4. 邨先棡縺ｮ邨ｱ蜷・
            complete_results = {
                'system_info': {
                    'name': 'Asia-Pacific Fire Monitoring and Anomaly Reasoning System',
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
            
            # 邨先棡菫晏ｭ・
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/complete_anomaly_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"笨・螳悟・蛻・梵邨先棡菫晏ｭ・ {output_file}")
            self.logger.info("脂 Asia-Pacific Fire Monitoring and Anomaly Analysis v3.4 螳御ｺ・)
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"笶・螳悟・蛻・梵繧ｨ繝ｩ繝ｼ: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # 繧ｷ繧ｹ繝・Β縺ｮ螳溯｡御ｾ・
    print("櫨 Asia-Pacific Fire Monitoring and Anomaly Reasoning System v3.4")
    print("=" * 60)
    
    # 繧ｷ繧ｹ繝・Β蛻晄悄蛹・
    system = GlobalFireMonitoringAndAnomalyReasoningSystemV33()
    
    # 螳悟・蛻・梵螳溯｡・
    results = system.run_complete_anomaly_analysis(year=2022, top_anomalies=15)
    
    if 'error' not in results:
        print(f"\n笨・蛻・梵螳御ｺ・")
        print(f"投 蜃ｦ逅・げ繝ｪ繝・ラ謨ｰ: {results['performance_metrics']['total_grids_processed']:,}")
        print(f"圷 逡ｰ蟶ｸ讀懃衍謨ｰ: {results['performance_metrics']['anomalies_detected']:,}")
        print(f"嶋 逡ｰ蟶ｸ邇・ {results['performance_metrics']['anomaly_rate']:.1%}")
        print(f"､・謗ｨ隲門ｮ御ｺ・焚: {results['performance_metrics']['reasoning_completed']}")
    else:
        print(f"笶・繧ｨ繝ｩ繝ｼ: {results['error']}")
