"""
QUICK START - Run individual experiments
"""

import sys
import os

def print_menu():
    """Print experiment menu"""
    print("\n" + "="*80)
    print("🔬 RESEARCH LAB - QUICK START")
    print("="*80)
    print("\nAvailable Experiments:")
    print("\n📁 REGIME DISCOVERY:")
    print("  1. HMM-based regime discovery")
    print("  2. Markov Switching Model")
    print("  3. K-Means clustering")
    
    print("\n📁 GAN ARCHITECTURES:")
    print("  4. TimeGAN (state-of-the-art)")
    print("  5. WGAN-GP (fixed version)")
    print("  6. RCGAN (recurrent)")
    
    print("\n📁 CLASSIFICATION MODELS:")
    print("  7. Transformer classifier")
    print("  8. TCN (Temporal Convolutional Network)")
    print("  9. GRU classifier")
    
    print("\n📁 LOSS FUNCTIONS:")
    print("  10. Focal Loss variants")
    print("  11. Dice Loss")
    print("  12. LDAM Loss")
    
    print("\n📁 MULTI-MARKET:")
    print("  13. Multi-market validation (6 markets)")
    
    print("\n📁 BASELINES:")
    print("  14. HMM baseline")
    print("  15. Random Forest baseline")
    
    print("\n📁 RUN ALL:")
    print("  99. Run ALL experiments")
    
    print("\n  0. Exit")
    print("="*80)

def run_experiment(choice):
    """Run selected experiment"""
    
    if choice == 1:
        print("\n🔬 Running HMM regime discovery...")
        from Labs-2.01_regime_discovery import hmm_regimes
        result = hmm_regimes.run_experiment()
        print(f"\n✅ Result: {result}")
        
    elif choice == 4:
        print("\n🔬 Running TimeGAN experiment...")
        from Labs-2.02_gan_architectures import timegan_experiment
        result = timegan_experiment.run_experiment()
        print(f"\n✅ Result: {result}")
        
    elif choice == 7:
        print("\n🔬 Running Transformer classifier...")
        from Labs-2.03_classification_models import transformer_classifier
        result = transformer_classifier.run_experiment()
        print(f"\n✅ Result: {result}")
        
    elif choice == 99:
        print("\n🔬 Running ALL experiments...")
        import run_all_experiments
        # This will run everything
        
    else:
        print(f"⚠️  Experiment {choice} not yet implemented")
        print("   Available: 1 (HMM), 4 (TimeGAN), 7 (Transformer), 99 (All)")

def main():
    """Main menu loop"""
    while True:
        print_menu()
        
        try:
            choice = int(input("\nSelect experiment (0-99): "))
            
            if choice == 0:
                print("\n👋 Exiting lab...")
                break
            
            run_experiment(choice)
            
            input("\n⏸️  Press Enter to continue...")
            
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting lab...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()

