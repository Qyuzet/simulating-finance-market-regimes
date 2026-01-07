"""
ENHANCED CONFORMAL PREDICTION FOR REGIME CLASSIFICATION
Addresses open research questions from Consensus AI (2025):
- Q2: Class-conditional CP for handling class imbalance
- Q3: Adaptive CP for non-stationary time series
"""

import numpy as np
import pandas as pd
from collections import deque

class ConformalRegimeClassifier:
    """
    Standard Conformal Prediction for Regime Classification
    Provides prediction SETS with guaranteed coverage (e.g., 90%)
    """
    
    def __init__(self, base_model, alpha=0.1):
        """
        Args:
            base_model: Trained classifier (e.g., GradientBoostingClassifier)
            alpha: Miscoverage rate (0.1 = 90% coverage guarantee)
        """
        self.model = base_model
        self.alpha = alpha
        self.q_level = None
        self.regime_names = ['Bear', 'Bull', 'Neutral']
        
    def calibrate(self, X_cal, y_cal):
        """
        Calibrate conformal predictor on held-out calibration set
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        # Get prediction probabilities
        probs = self.model.predict_proba(X_cal)
        
        # Compute conformity scores (1 - probability of true class)
        conformity_scores = []
        for i, true_label in enumerate(y_cal):
            score = 1 - probs[i, true_label]
            conformity_scores.append(score)
        
        conformity_scores = np.array(conformity_scores)
        
        # Find quantile for desired coverage
        n = len(conformity_scores)
        q_level_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q_level = np.sort(conformity_scores)[min(q_level_idx - 1, n - 1)]
        
    def predict_set(self, X_test):
        """
        Return prediction sets with guaranteed coverage
        
        Args:
            X_test: Test features
            
        Returns:
            prediction_sets: List of sets, each containing regime indices
        """
        probs = self.model.predict_proba(X_test)
        
        prediction_sets = []
        for prob in probs:
            # Include all regimes with (1 - prob) <= q_level
            pred_set = [i for i in range(len(prob)) if (1 - prob[i]) <= self.q_level]
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def predict_point(self, X_test):
        """Standard point prediction (for comparison)"""
        return self.model.predict(X_test)
    
    def evaluate_coverage(self, X_test, y_test):
        """
        Evaluate empirical coverage on test set

        Returns:
            coverage: Fraction of test samples where true label is in prediction set
            avg_set_size: Average size of prediction sets
        """
        pred_sets = self.predict_set(X_test)

        # Check coverage - FIX: properly check if true label is in prediction set
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        covered = [(y_test_array[i] in pred_set) for i, pred_set in enumerate(pred_sets)]
        coverage = np.mean(covered)

        # Average set size
        set_sizes = [len(pred_set) for pred_set in pred_sets]
        avg_set_size = np.mean(set_sizes)

        # Set size distribution
        set_size_dist = {i: set_sizes.count(i) for i in range(0, 4)}

        # Per-class coverage
        per_class_coverage = {}
        for class_idx in range(3):
            class_mask = y_test_array == class_idx
            if class_mask.sum() > 0:
                class_covered = [covered[i] for i in range(len(covered)) if class_mask[i]]
                per_class_coverage[class_idx] = np.mean(class_covered)

        return coverage, avg_set_size, set_size_dist, pred_sets, per_class_coverage


class ClassConditionalCP(ConformalRegimeClassifier):
    """
    Q2 ANSWER: Class-Conditional Conformal Prediction
    
    Addresses class imbalance by using separate quantiles for each class.
    This ensures balanced coverage across all regimes (Bear, Bull, Neutral).
    
    Reference: Guan & Tibshirani (2019), Liu & Ong (2025)
    """
    
    def __init__(self, base_model, alpha=0.1, class_weights=None):
        """
        Args:
            base_model: Trained classifier
            alpha: Miscoverage rate (can be dict for per-class alpha)
            class_weights: Optional weights for each class (for weighted conformity scores)
        """
        super().__init__(base_model, alpha)
        self.q_levels = {}  # Separate quantile for each class
        self.class_weights = class_weights
        
    def calibrate(self, X_cal, y_cal):
        """
        Calibrate with class-conditional quantiles
        """
        probs = self.model.predict_proba(X_cal)
        y_cal_array = y_cal.values if hasattr(y_cal, 'values') else y_cal

        # Compute conformity scores per class
        for class_idx in range(3):
            # Get samples from this class
            class_mask = y_cal_array == class_idx
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Compute conformity scores for this class
            conformity_scores = []
            for i in class_indices:
                score = 1 - probs[i, class_idx]
                # Apply class weight if provided
                if self.class_weights is not None:
                    score *= self.class_weights[class_idx]
                conformity_scores.append(score)

            conformity_scores = np.array(conformity_scores)

            # Compute class-specific quantile
            n = len(conformity_scores)
            alpha_class = self.alpha[class_idx] if isinstance(self.alpha, dict) else self.alpha
            q_level_idx = int(np.ceil((n + 1) * (1 - alpha_class)))
            self.q_levels[class_idx] = np.sort(conformity_scores)[min(q_level_idx - 1, n - 1)]

    def predict_set(self, X_test):
        """
        Return prediction sets using class-conditional quantiles
        """
        probs = self.model.predict_proba(X_test)

        prediction_sets = []
        for prob in probs:
            pred_set = []
            for class_idx in range(len(prob)):
                if class_idx in self.q_levels:
                    score = 1 - prob[class_idx]
                    if self.class_weights is not None:
                        score *= self.class_weights[class_idx]
                    if score <= self.q_levels[class_idx]:
                        pred_set.append(class_idx)
            prediction_sets.append(pred_set)

        return prediction_sets


class AdaptiveCP(ConformalRegimeClassifier):
    """
    Q3 ANSWER: Adaptive Conformal Prediction for Non-Stationary Time Series

    Uses sliding window recalibration to handle regime shifts and non-stationarity.
    Dynamically updates quantiles as new data arrives.

    Reference: Aich et al. (2025) - Temporal Conformal Prediction (TCP)
              Gibbs & Candès (2022) - Online CP with distribution shifts
    """

    def __init__(self, base_model, alpha=0.1, window_size=252):
        """
        Args:
            base_model: Trained classifier
            alpha: Miscoverage rate
            window_size: Size of sliding window for recalibration (default: 252 = 1 year)
        """
        super().__init__(base_model, alpha)
        self.window_size = window_size
        self.conformity_scores_buffer = deque(maxlen=window_size)
        self.is_calibrated = False

    def calibrate(self, X_cal, y_cal):
        """
        Initial calibration on calibration set
        """
        probs = self.model.predict_proba(X_cal)
        y_cal_array = y_cal.values if hasattr(y_cal, 'values') else y_cal

        # Compute conformity scores
        for i, true_label in enumerate(y_cal_array):
            score = 1 - probs[i, true_label]
            self.conformity_scores_buffer.append(score)

        self._update_quantile()
        self.is_calibrated = True

    def _update_quantile(self):
        """
        Update quantile based on current buffer
        """
        if len(self.conformity_scores_buffer) == 0:
            return

        scores = np.array(list(self.conformity_scores_buffer))
        n = len(scores)
        q_level_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q_level = np.sort(scores)[min(q_level_idx - 1, n - 1)]

    def update(self, X_new, y_new):
        """
        Online update: Add new observation and recalibrate

        Args:
            X_new: New feature vector (single sample or batch)
            y_new: True label(s)
        """
        # Handle single sample or batch
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)
            y_new = [y_new]

        probs = self.model.predict_proba(X_new)
        y_new_array = y_new.values if hasattr(y_new, 'values') else y_new

        # Add new conformity scores to buffer
        for i, true_label in enumerate(y_new_array):
            score = 1 - probs[i, true_label]
            self.conformity_scores_buffer.append(score)

        # Recalibrate with updated buffer
        self._update_quantile()

    def predict_set_online(self, X_test, y_test=None, update=True):
        """
        Online prediction with optional recalibration

        Args:
            X_test: Test features
            y_test: True labels (optional, for online update)
            update: Whether to update quantile after each prediction

        Returns:
            prediction_sets: List of prediction sets
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before prediction")

        prediction_sets = []

        for i in range(len(X_test)):
            X_i = X_test.iloc[[i]] if hasattr(X_test, 'iloc') else X_test[i:i+1]

            # Make prediction with current quantile
            pred_set = self.predict_set(X_i)[0]
            prediction_sets.append(pred_set)

            # Update if true label is provided
            if y_test is not None and update:
                y_i = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                self.update(X_i, [y_i])

        return prediction_sets


class AdaptiveClassConditionalCP(ConformalRegimeClassifier):
    """
    HYBRID METHOD: Adaptive + Class-Conditional Conformal Prediction

    Combines:
    - Adaptive calibration (Q3): Handles non-stationarity with sliding window
    - Class-conditional quantiles (Q2): Handles class imbalance with per-class calibration

    This is the STRONGEST method, addressing both Q2 and Q3 simultaneously.

    Reference: Combines techniques from:
              - Guan & Tibshirani (2019) - Class-conditional CP
              - Gibbs & Candès (2022) - Online CP with distribution shifts
    """

    def __init__(self, base_model, alpha=0.1, window_size=252, class_weights=None):
        """
        Args:
            base_model: Trained classifier
            alpha: Miscoverage rate
            window_size: Size of sliding window for recalibration (default: 252 = 1 year)
            class_weights: Optional dict of class weights (if None, uses inverse frequency)
        """
        super().__init__(base_model, alpha)
        self.window_size = window_size
        self.class_weights = class_weights
        self.conformity_scores_buffers = {}  # Separate buffer per class
        self.q_levels = {}  # Separate quantile per class
        self.is_calibrated = False

    def calibrate(self, X_cal, y_cal):
        """
        Calibrate with class-conditional sliding windows
        """
        probs = self.model.predict_proba(X_cal)
        y_cal_array = y_cal.values if hasattr(y_cal, 'values') else y_cal

        # Compute class weights if not provided (inverse frequency)
        if self.class_weights is None:
            unique_classes, class_counts = np.unique(y_cal_array, return_counts=True)
            n_samples = len(y_cal_array)
            n_classes = len(unique_classes)
            self.class_weights = {
                int(cls): n_samples / (n_classes * count)
                for cls, count in zip(unique_classes, class_counts)
            }

        # Initialize buffers for each class
        unique_classes = np.unique(y_cal_array)
        for cls in unique_classes:
            self.conformity_scores_buffers[int(cls)] = deque(maxlen=self.window_size)

        # Compute weighted conformity scores per class
        for i, true_label in enumerate(y_cal_array):
            score = 1 - probs[i, true_label]
            weighted_score = score * self.class_weights[int(true_label)]
            self.conformity_scores_buffers[int(true_label)].append(weighted_score)

        # Compute quantile for each class
        self._update_quantiles()
        self.is_calibrated = True

    def _update_quantiles(self):
        """Update quantiles for all classes"""
        for cls in self.conformity_scores_buffers.keys():
            if len(self.conformity_scores_buffers[cls]) > 0:
                scores = np.array(list(self.conformity_scores_buffers[cls]))
                n = len(scores)
                q_level_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
                self.q_levels[cls] = np.sort(scores)[min(q_level_idx - 1, n - 1)]
            else:
                self.q_levels[cls] = 0.0

    def update(self, X_new, y_new):
        """
        Online update: Add new observation to class-specific buffer and recalibrate

        Args:
            X_new: New observation features (single sample or batch)
            y_new: True labels (single label or array)
        """
        probs = self.model.predict_proba(X_new)
        y_new_array = y_new if isinstance(y_new, (list, np.ndarray)) else [y_new]

        for i, true_label in enumerate(y_new_array):
            score = 1 - probs[i, true_label]
            weighted_score = score * self.class_weights.get(int(true_label), 1.0)

            # Add to class-specific buffer
            if int(true_label) not in self.conformity_scores_buffers:
                self.conformity_scores_buffers[int(true_label)] = deque(maxlen=self.window_size)
            self.conformity_scores_buffers[int(true_label)].append(weighted_score)

        # Recalculate quantiles
        self._update_quantiles()

    def predict_set(self, X_test):
        """
        Predict using class-conditional quantiles
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before prediction")

        probs = self.model.predict_proba(X_test)
        prediction_sets = []

        for prob in probs:
            pred_set = []
            for class_idx in range(len(prob)):
                if class_idx in self.q_levels:
                    score = 1 - prob[class_idx]
                    weighted_score = score * self.class_weights.get(class_idx, 1.0)
                    if weighted_score <= self.q_levels[class_idx]:
                        pred_set.append(class_idx)
            prediction_sets.append(pred_set)

        return prediction_sets

    def predict_set_online(self, X_test, y_test=None, update=True):
        """
        Online prediction with class-conditional adaptive calibration

        Args:
            X_test: Test features
            y_test: True labels (optional, for updating calibration)
            update: Whether to update calibration after each prediction
        """
        prediction_sets = []

        for i in range(len(X_test)):
            X_i = X_test[i:i+1]
            pred_set = self.predict_set(X_i)[0]
            prediction_sets.append(pred_set)

            # Update class-specific calibration if labels provided
            if y_test is not None and update:
                y_i = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                self.update(X_i, [y_i])

        return prediction_sets

