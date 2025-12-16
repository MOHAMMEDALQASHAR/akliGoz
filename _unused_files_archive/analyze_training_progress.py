# -*- coding: utf-8 -*-
"""
تحليل تقدم التدريب Epoch بـ Epoch
Training Progress Analysis - Epoch by Epoch
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_training_progress():
    print("=" * 70)
    print("تحليل تقدم التدريب - Training Progress Analysis")
    print("=" * 70)
    
    results_csv = "runs/classify/currency_cls_advanced/results.csv"
    
    if not os.path.exists(results_csv):
        print(f"\n❌ Results file not found: {results_csv}")
        return
    
    # Read CSV
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names
        
        print(f"\n✅ Loaded training results")
        print(f"📊 Total Epochs: {len(df)}")
        
        # Display available columns
        print(f"\n📋 Available Metrics: {', '.join(df.columns.tolist())}")
        
        # Show epoch-by-epoch progress
        print("\n" + "=" * 70)
        print("📈 Epoch-by-Epoch Progress")
        print("=" * 70)
        
        # Find accuracy columns
        acc_cols = [col for col in df.columns if 'acc' in col.lower() or 'top' in col.lower()]
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        
        # Create formatted table header
        print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Accuracy':<15} {'Val Top-5':<12}")
        print("-" * 70)
        
        # Print key epochs (first, every 10th, and last)
        key_epochs = []
        for i in range(len(df)):
            if i == 0 or i % 10 == 9 or i == len(df) - 1:
                key_epochs.append(i)
        
        for idx in key_epochs:
            row = df.iloc[idx]
            epoch = idx + 1
            
            # Extract values (handle different possible column names)
            train_loss = row.get('train/loss', row.get('loss', 'N/A'))
            val_loss = row.get('val/loss', row.get('val_loss', 'N/A'))
            
            # Try different accuracy column names
            val_acc = None
            for col in ['metrics/accuracy_top1', 'val/acc', 'accuracy', 'top1_acc']:
                if col in df.columns:
                    val_acc = row.get(col)
                    break
            
            val_top5 = None
            for col in ['metrics/accuracy_top5', 'top5_acc']:
                if col in df.columns:
                    val_top5 = row.get(col)
                    break
            
            # Format values
            train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
            val_acc_str = f"{val_acc*100:.2f}%" if val_acc is not None else "N/A"
            val_top5_str = f"{val_top5*100:.2f}%" if val_top5 is not None else "N/A"
            
            print(f"{epoch:<8} {train_loss_str:<12} {val_loss_str:<12} {val_acc_str:<15} {val_top5_str:<12}")
        
        # Final results
        print("\n" + "=" * 70)
        print("🎯 FINAL RESULTS (Last Epoch)")
        print("=" * 70)
        
        last_row = df.iloc[-1]
        
        # Get all metrics from last epoch
        print(f"\n📊 Epoch {len(df)} Performance:")
        print("-" * 40)
        
        for col in df.columns:
            if col.strip() == 'epoch':
                continue
            value = last_row[col]
            
            # Format percentages
            if 'acc' in col.lower() or 'top' in col.lower():
                if isinstance(value, (int, float)) and value <= 1.0:
                    print(f"   {col:<30} {value*100:.2f}%")
                else:
                    print(f"   {col:<30} {value}")
            elif 'loss' in col.lower():
                print(f"   {col:<30} {value:.4f}")
            else:
                print(f"   {col:<30} {value}")
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("📈 Training Summary")
        print("=" * 70)
        
        # Best epoch
        if val_acc is not None:
            acc_col = None
            for col in ['metrics/accuracy_top1', 'val/acc', 'accuracy', 'top1_acc']:
                if col in df.columns:
                    acc_col = col
                    break
            
            if acc_col:
                best_epoch = df[acc_col].idxmax() + 1
                best_acc = df[acc_col].max()
                print(f"\n🏆 Best Validation Accuracy:")
                print(f"   Epoch: {best_epoch}")
                print(f"   Accuracy: {best_acc*100:.2f}%")
                
                # When did it reach 95%, 98%, 99%?
                milestones = [0.95, 0.98, 0.99, 1.00]
                print(f"\n🎯 Accuracy Milestones:")
                print("-" * 40)
                for milestone in milestones:
                    reached = df[df[acc_col] >= milestone]
                    if not reached.empty:
                        first_epoch = reached.index[0] + 1
                        print(f"   {milestone*100:.0f}%: Reached at Epoch {first_epoch}")
                    else:
                        print(f"   {milestone*100:.0f}%: Not reached")
        
        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)
        
        # Try to create a simple plot
        try:
            print("\n📊 Generating training plot...")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Loss
            if 'train/loss' in df.columns and 'val/loss' in df.columns:
                axes[0].plot(df.index + 1, df['train/loss'], label='Train Loss', linewidth=2)
                axes[0].plot(df.index + 1, df['val/loss'], label='Val Loss', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training & Validation Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Accuracy
            if acc_col:
                axes[1].plot(df.index + 1, df[acc_col] * 100, label='Val Top-1 Accuracy', 
                           linewidth=2, color='green')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy (%)')
                axes[1].set_title('Validation Accuracy Progress')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim([0, 105])
            
            plt.tight_layout()
            output_path = "training_progress.png"
            plt.savefig(output_path, dpi=150)
            print(f"✅ Plot saved to: {output_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")
        
    except Exception as e:
        print(f"\n❌ Error reading results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_training_progress()
