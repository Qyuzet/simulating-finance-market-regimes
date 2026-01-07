"""
MASTER EXPERIMENT RUNNER
Run ALL experiments across all labs and generate comparison tables
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ExperimentRunner:
    def __init__(self):
        self.results_dir = "Labs-2/results"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        
        self.experiment_log = []
        self.log_file = f"{self.results_dir}/experiment_log.csv"
        
    def log_result(self, experiment_name, category, metrics, notes=""):
        """Log experiment result"""
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment': experiment_name,
            'category': category,
            'notes': notes,
            **metrics
        }
        self.experiment_log.append(result)
        
        # Save incrementally
        df = pd.DataFrame(self.experiment_log)
        df.to_csv(self.log_file, index=False)
        print(f"✅ Logged: {experiment_name}")
        
    def run_regime_discovery_experiments(self):
        """Run all regime discovery experiments"""
        print("\n" + "="*80)
        print("🔬 LAB 01: REGIME DISCOVERY METHODS")
        print("="*80)
        
        # Import and run each experiment
        try:
            from Labs-2.01_regime_discovery import hmm_regimes
            result = hmm_regimes.run_experiment()
            self.log_result("HMM Regime Discovery", "regime_discovery", result)
        except Exception as e:
            print(f"❌ HMM failed: {e}")
            
        try:
            from Labs-2.01_regime_discovery import markov_switching
            result = markov_switching.run_experiment()
            self.log_result("Markov Switching", "regime_discovery", result)
        except Exception as e:
            print(f"❌ Markov Switching failed: {e}")
            
        try:
            from Labs-2.01_regime_discovery import kmeans_regimes
            result = kmeans_regimes.run_experiment()
            self.log_result("K-Means Clustering", "regime_discovery", result)
        except Exception as e:
            print(f"❌ K-Means failed: {e}")
            
    def run_gan_experiments(self):
        """Run all GAN architecture experiments"""
        print("\n" + "="*80)
        print("🔬 LAB 02: GAN ARCHITECTURES")
        print("="*80)
        
        try:
            from Labs-2.02_gan_architectures import timegan_experiment
            result = timegan_experiment.run_experiment()
            self.log_result("TimeGAN", "gan_architecture", result)
        except Exception as e:
            print(f"❌ TimeGAN failed: {e}")
            
        try:
            from Labs-2.02_gan_architectures import wgan_gp_fixed
            result = wgan_gp_fixed.run_experiment()
            self.log_result("WGAN-GP (Fixed)", "gan_architecture", result)
        except Exception as e:
            print(f"❌ WGAN-GP failed: {e}")
            
        try:
            from Labs-2.02_gan_architectures import rcgan_experiment
            result = rcgan_experiment.run_experiment()
            self.log_result("RCGAN", "gan_architecture", result)
        except Exception as e:
            print(f"❌ RCGAN failed: {e}")
            
    def run_classification_experiments(self):
        """Run all classification model experiments"""
        print("\n" + "="*80)
        print("🔬 LAB 03: CLASSIFICATION MODELS")
        print("="*80)
        
        try:
            from Labs-2.03_classification_models import transformer_classifier
            result = transformer_classifier.run_experiment()
            self.log_result("Transformer", "classification", result)
        except Exception as e:
            print(f"❌ Transformer failed: {e}")
            
        try:
            from Labs-2.03_classification_models import tcn_classifier
            result = tcn_classifier.run_experiment()
            self.log_result("TCN", "classification", result)
        except Exception as e:
            print(f"❌ TCN failed: {e}")
            
    def run_loss_function_experiments(self):
        """Run all loss function experiments"""
        print("\n" + "="*80)
        print("🔬 LAB 04: LOSS FUNCTIONS")
        print("="*80)
        
        try:
            from Labs-2.04_loss_functions import focal_loss_variants
            result = focal_loss_variants.run_experiment()
            self.log_result("Focal Loss Variants", "loss_function", result)
        except Exception as e:
            print(f"❌ Focal Loss variants failed: {e}")
            
    def run_multi_market_experiments(self):
        """Run multi-market validation"""
        print("\n" + "="*80)
        print("🔬 LAB 05: MULTI-MARKET VALIDATION")
        print("="*80)
        
        try:
            from Labs-2.05_multi_market import multi_market_validation
            result = multi_market_validation.run_experiment()
            self.log_result("Multi-Market Validation", "multi_market", result)
        except Exception as e:
            print(f"❌ Multi-market failed: {e}")
            
    def run_baseline_experiments(self):
        """Run SOTA baseline comparisons"""
        print("\n" + "="*80)
        print("🔬 LAB 06: SOTA BASELINES")
        print("="*80)
        
        try:
            from Labs-2.06_baselines import hmm_baseline
            result = hmm_baseline.run_experiment()
            self.log_result("HMM Baseline", "baseline", result)
        except Exception as e:
            print(f"❌ HMM baseline failed: {e}")
            
    def generate_comparison_tables(self):
        """Generate comparison tables from all results"""
        print("\n" + "="*80)
        print("📊 GENERATING COMPARISON TABLES")
        print("="*80)
        
        df = pd.DataFrame(self.experiment_log)
        
        # Save detailed results
        df.to_csv(self.log_file, index=False)
        print(f"✅ Saved: {self.log_file}")
        
        # Generate summary tables by category
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            print(f"\n📋 {category.upper()} RESULTS:")
            print(category_df.to_string())
            
if __name__ == "__main__":
    runner = ExperimentRunner()
    
    print("🚀 STARTING ALL EXPERIMENTS")
    print("="*80)
    
    # Run all experiments
    runner.run_regime_discovery_experiments()
    runner.run_gan_experiments()
    runner.run_classification_experiments()
    runner.run_loss_function_experiments()
    runner.run_multi_market_experiments()
    runner.run_baseline_experiments()
    
    # Generate final comparison
    runner.generate_comparison_tables()
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"📊 Results saved to: {runner.results_dir}/")

