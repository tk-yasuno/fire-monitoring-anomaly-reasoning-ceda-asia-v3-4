"""
Text Processor for Africa Fire Report Analysis Pipeline v2.0

This module provides advanced text preprocessing and feature extraction
specifically designed for fire alert text data from Global Forest Watch.
"""

import re
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import string

# NLP libraries with graceful fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class FireAlertTextProcessor:
    """
    Advanced text processor for fire alert data.
    
    Provides comprehensive text preprocessing, feature extraction,
    and linguistic analysis specifically for fire alert texts.
    """
    
    def __init__(self, language: str = "english", logger: Optional[logging.Logger] = None):
        """
        Initialize text processor.
        
        Args:
            language (str): Primary language for text processing
            logger (Optional[logging.Logger]): Custom logger instance
        """
        self.language = language
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize NLP components
        self._setup_nltk()
        
        # Fire-specific vocabulary and patterns
        self.fire_vocabulary = {
            'fire_terms': [
                'fire', 'burn', 'burning', 'burnt', 'smoke', 'flame', 'blaze',
                'wildfire', 'forest fire', 'grass fire', 'bush fire',
                'charred', 'incinerate', 'combustion', 'ignition',
                'smolder', 'ember', 'ash', 'soot'
            ],
            'detection_terms': [
                'alert', 'detection', 'detected', 'identified', 'spotted',
                'observed', 'reported', 'confirmed', 'thermal', 'signature',
                'anomaly', 'hot spot', 'hotspot', 'heat signature'
            ],
            'intensity_terms': [
                'high', 'medium', 'low', 'intense', 'severe', 'moderate',
                'light', 'heavy', 'strong', 'weak', 'major', 'minor'
            ],
            'vegetation_terms': [
                'forest', 'woodland', 'grassland', 'savanna', 'vegetation',
                'trees', 'grass', 'bush', 'shrub', 'canopy', 'understory',
                'fynbos', 'miombo', 'acacia', 'eucalyptus', 'pine'
            ],
            'location_terms': [
                'region', 'area', 'zone', 'district', 'province', 'state',
                'county', 'territory', 'national park', 'reserve', 'conservation'
            ]
        }
        
        # Coordinate and numeric patterns
        self.coordinate_pattern = re.compile(r'(-?\d+\.?\d*)[°\s]*([NS]?)[\s,]*(-?\d+\.?\d*)[°\s]*([EW]?)')
        self.temperature_pattern = re.compile(r'(\d+\.?\d*)\s*[°]?[CF]')
        self.area_pattern = re.compile(r'(\d+\.?\d*)\s*(hectare|hectares|ha|km²|square\s*km|acre|acres)')
        self.confidence_pattern = re.compile(r'(\d+\.?\d*)\s*%|\b(high|medium|low)\s*confidence')
        
        # African country names for detection
        self.african_countries = [
            'Nigeria', 'Kenya', 'Ghana', 'Ethiopia', 'South Africa', 'Tanzania',
            'Uganda', 'Rwanda', 'Burundi', 'Somalia', 'Djibouti', 'Eritrea',
            'South Sudan', 'Sudan', 'Egypt', 'Libya', 'Tunisia', 'Algeria',
            'Morocco', 'Chad', 'Niger', 'Mali', 'Burkina Faso', 'Senegal',
            'Gambia', 'Guinea-Bissau', 'Guinea', 'Sierra Leone', 'Liberia',
            'Côte d\'Ivoire', 'Togo', 'Benin', 'Mauritania', 'Cape Verde',
            'São Tomé and Príncipe', 'Equatorial Guinea', 'Gabon',
            'Republic of Congo', 'Democratic Republic of Congo',
            'Central African Republic', 'Cameroon', 'Angola', 'Zambia',
            'Malawi', 'Mozambique', 'Zimbabwe', 'Botswana', 'Namibia',
            'Lesotho', 'Eswatini'
        ]
    
    def _setup_nltk(self):
        """Setup NLTK components with graceful fallback."""
        if not NLTK_AVAILABLE:
            self.logger.warning("NLTK not available. Some features will be limited.")
            self.stop_words = set()
            self.stemmer = None
            return
        
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Setup components
            self.stop_words = set(stopwords.words(self.language))
            self.stemmer = PorterStemmer()
            
        except Exception as e:
            self.logger.warning(f"NLTK setup failed: {e}. Using fallback methods.")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.stemmer = None
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess fire alert texts.
        
        Args:
            texts (List[str]): List of raw text descriptions
            
        Returns:
            List[str]: Preprocessed and cleaned texts
        """
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                processed_texts.append("")
                continue
            
            # Step 1: Basic cleaning
            cleaned_text = self._basic_cleaning(text)
            
            # Step 2: Fire-specific normalization
            normalized_text = self._normalize_fire_terminology(cleaned_text)
            
            # Step 3: Extract and normalize numeric entities
            processed_text = self._normalize_numeric_entities(normalized_text)
            
            # Step 4: Language detection and filtering
            if self._is_valid_language(processed_text):
                processed_texts.append(processed_text)
            else:
                processed_texts.append("")
        
        return processed_texts
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,;:°%\-]', ' ', text)
        
        # Fix common OCR/encoding issues
        text = text.replace('â', 'a').replace('ã', 'a').replace('ô', 'o')
        
        return text.strip()
    
    def _normalize_fire_terminology(self, text: str) -> str:
        """Normalize fire-specific terminology."""
        # Standardize fire terms
        text = re.sub(r'\bwild\s*fire\b', 'wildfire', text)
        text = re.sub(r'\bforest\s*fire\b', 'forest fire', text)
        text = re.sub(r'\bgrass\s*fire\b', 'grass fire', text)
        text = re.sub(r'\bbush\s*fire\b', 'bush fire', text)
        text = re.sub(r'\bhot\s*spot\b', 'hotspot', text)
        
        # Standardize intensity terms
        text = re.sub(r'\bhigh\s*intensity\b', 'high intensity', text)
        text = re.sub(r'\blow\s*intensity\b', 'low intensity', text)
        
        # Standardize confidence terms
        text = re.sub(r'\bhigh\s*confidence\b', 'high confidence', text)
        text = re.sub(r'\bmedium\s*confidence\b', 'medium confidence', text)
        text = re.sub(r'\blow\s*confidence\b', 'low confidence', text)
        
        return text
    
    def _normalize_numeric_entities(self, text: str) -> str:
        """Normalize numeric entities in text."""
        # Normalize temperature mentions
        text = re.sub(r'(\d+\.?\d*)\s*degrees?\s*celsius', r'\1°C', text)
        text = re.sub(r'(\d+\.?\d*)\s*degrees?\s*fahrenheit', r'\1°F', text)
        
        # Normalize area mentions
        text = re.sub(r'(\d+\.?\d*)\s*hectares?', r'\1 hectares', text)
        text = re.sub(r'(\d+\.?\d*)\s*ha\b', r'\1 hectares', text)
        
        # Normalize confidence percentages
        text = re.sub(r'(\d+\.?\d*)\s*percent', r'\1%', text)
        
        return text
    
    def _is_valid_language(self, text: str) -> bool:
        """Check if text is in a valid language."""
        if not LANGDETECT_AVAILABLE or len(text.split()) < 3:
            return True  # Accept all if can't detect or too short
        
        try:
            detected_lang = detect(text)
            # Accept English, French (common in Africa), and uncertain cases
            return detected_lang in ['en', 'fr'] or len(text.split()) < 5
        except:
            return True  # Accept if detection fails
    
    def extract_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features from processed texts.
        
        Args:
            texts (List[str]): Preprocessed texts
            
        Returns:
            Dict[str, np.ndarray]: Feature matrices
        """
        features = {}
        
        # Linguistic features
        features['linguistic'] = self._extract_linguistic_features(texts)
        
        # Fire-specific categorical features
        features['categorical'] = self._extract_categorical_features(texts)
        
        # Numeric entity features
        features['numeric'] = self._extract_numeric_features(texts)
        
        # Text statistics
        features['statistics'] = self._extract_text_statistics(texts)
        
        return features
    
    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts."""
        features = []
        
        for text in texts:
            if not text:
                features.append([0] * 10)  # Zero features for empty text
                continue
            
            # Basic linguistic features
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(self._split_sentences(text))
            avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Fire vocabulary density
            fire_term_count = sum(1 for term in self.fire_vocabulary['fire_terms'] if term in text)
            fire_density = fire_term_count / word_count if word_count > 0 else 0
            
            # Detection term density
            detection_count = sum(1 for term in self.fire_vocabulary['detection_terms'] if term in text)
            detection_density = detection_count / word_count if word_count > 0 else 0
            
            # Readability approximation (Flesch-like)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            readability_score = max(0, min(100, readability_score))  # Clamp to 0-100
            
            # Complexity measures
            unique_words = len(set(text.split()))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            
            features.append([
                word_count, char_count, sentence_count, avg_word_length,
                fire_density, detection_density, readability_score,
                lexical_diversity, avg_sentence_length, unique_words
            ])
        
        return np.array(features)
    
    def _extract_categorical_features(self, texts: List[str]) -> np.ndarray:
        """Extract categorical features related to fire alerts."""
        features = []
        
        for text in texts:
            if not text:
                features.append([0] * 8)
                continue
            
            # Fire type indicators
            has_wildfire = 1 if 'wildfire' in text else 0
            has_forest_fire = 1 if 'forest fire' in text else 0
            has_grass_fire = 1 if 'grass fire' in text else 0
            has_bush_fire = 1 if 'bush fire' in text else 0
            
            # Intensity indicators
            has_high_intensity = 1 if any(term in text for term in ['high', 'intense', 'severe']) else 0
            has_low_intensity = 1 if any(term in text for term in ['low', 'light', 'minor']) else 0
            
            # Confidence indicators
            has_confidence_mention = 1 if 'confidence' in text else 0
            
            # Smoke indicators
            has_smoke_mention = 1 if 'smoke' in text else 0
            
            features.append([
                has_wildfire, has_forest_fire, has_grass_fire, has_bush_fire,
                has_high_intensity, has_low_intensity, has_confidence_mention,
                has_smoke_mention
            ])
        
        return np.array(features)
    
    def _extract_numeric_features(self, texts: List[str]) -> np.ndarray:
        """Extract numeric features from texts."""
        features = []
        
        for text in texts:
            if not text:
                features.append([0] * 6)
                continue
            
            # Extract coordinates
            coordinates = self.extract_numeric_entities(text).get('coordinates', [])
            has_coordinates = 1 if coordinates else 0
            
            # Extract temperatures
            temperatures = self.extract_numeric_entities(text).get('temperatures', [])
            max_temp = max(temperatures) if temperatures else 0
            
            # Extract areas
            areas = self.extract_numeric_entities(text).get('areas', [])
            max_area = max(areas) if areas else 0
            
            # Extract confidence scores
            confidence_scores = self.extract_numeric_entities(text).get('confidence_scores', [])
            max_confidence = max(confidence_scores) if confidence_scores else 0
            
            # Count numeric mentions
            numeric_count = len(re.findall(r'\d+\.?\d*', text))
            
            # Percentage mentions
            percentage_count = len(re.findall(r'\d+\.?\d*\s*%', text))
            
            features.append([
                has_coordinates, max_temp, max_area, max_confidence,
                numeric_count, percentage_count
            ])
        
        return np.array(features)
    
    def _extract_text_statistics(self, texts: List[str]) -> np.ndarray:
        """Extract basic text statistics."""
        features = []
        
        for text in texts:
            if not text:
                features.append([0] * 5)
                continue
            
            # Basic statistics
            word_count = len(text.split())
            char_count = len(text)
            punctuation_count = sum(1 for char in text if char in string.punctuation)
            uppercase_count = sum(1 for char in text if char.isupper())
            digit_count = sum(1 for char in text if char.isdigit())
            
            features.append([
                word_count, char_count, punctuation_count,
                uppercase_count, digit_count
            ])
        
        return np.array(features)
    
    def extract_numeric_entities(self, text: str) -> Dict[str, List[float]]:
        """
        Extract numeric entities from fire alert text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[float]]: Extracted numeric entities
        """
        entities = {
            'coordinates': [],
            'temperatures': [],
            'areas': [],
            'confidence_scores': [],
            'distances': []
        }
        
        # Extract coordinates
        coord_matches = self.coordinate_pattern.findall(text)
        for match in coord_matches:
            lat, lat_dir, lon, lon_dir = match
            try:
                lat_val = float(lat)
                lon_val = float(lon)
                if lat_dir.upper() == 'S':
                    lat_val = -lat_val
                if lon_dir.upper() == 'W':
                    lon_val = -lon_val
                entities['coordinates'].extend([lat_val, lon_val])
            except ValueError:
                continue
        
        # Extract temperatures
        temp_matches = self.temperature_pattern.findall(text)
        for temp in temp_matches:
            try:
                entities['temperatures'].append(float(temp))
            except ValueError:
                continue
        
        # Extract areas
        area_matches = self.area_pattern.findall(text)
        for area, unit in area_matches:
            try:
                area_val = float(area)
                # Convert to hectares
                if 'km' in unit.lower() or 'square' in unit.lower():
                    area_val *= 100  # km² to hectares
                elif 'acre' in unit.lower():
                    area_val *= 0.4047  # acres to hectares
                entities['areas'].append(area_val)
            except ValueError:
                continue
        
        # Extract confidence scores
        conf_matches = self.confidence_pattern.findall(text)
        for conf in conf_matches:
            if conf[0]:  # Numeric confidence
                try:
                    entities['confidence_scores'].append(float(conf[0]))
                except ValueError:
                    continue
            elif conf[1]:  # Text confidence
                conf_map = {'high': 85, 'medium': 65, 'low': 45}
                entities['confidence_scores'].append(conf_map.get(conf[1].lower(), 50))
        
        return entities
    
    def detect_country(self, text: str) -> Optional[str]:
        """
        Detect country mentions in text.
        
        Args:
            text (str): Input text
            
        Returns:
            Optional[str]: Detected country name or None
        """
        text_lower = text.lower()
        
        for country in self.african_countries:
            if country.lower() in text_lower:
                return country
        
        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with fallback."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_text_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive text statistics for the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame with text data
            
        Returns:
            Dict[str, Any]: Text statistics
        """
        if 'alert_text' not in data.columns:
            return {}
        
        texts = data['alert_text'].fillna('').astype(str)
        
        # Basic statistics
        total_texts = len(texts)
        non_empty_texts = sum(1 for text in texts if text.strip())
        
        # Length statistics
        word_counts = [len(text.split()) for text in texts if text.strip()]
        char_counts = [len(text) for text in texts if text.strip()]
        
        # Vocabulary analysis
        all_words = []
        for text in texts:
            if text.strip():
                all_words.extend(text.lower().split())
        
        vocabulary_size = len(set(all_words))
        most_common_words = Counter(all_words).most_common(20)
        
        # Fire term analysis
        fire_term_counts = Counter()
        for term in self.fire_vocabulary['fire_terms']:
            count = sum(1 for text in texts if term in text.lower())
            if count > 0:
                fire_term_counts[term] = count
        
        return {
            'total_texts': total_texts,
            'non_empty_texts': non_empty_texts,
            'avg_word_count': np.mean(word_counts) if word_counts else 0,
            'avg_char_count': np.mean(char_counts) if char_counts else 0,
            'vocabulary_size': vocabulary_size,
            'most_common_words': most_common_words[:10],
            'fire_term_frequencies': dict(fire_term_counts.most_common(10))
        }