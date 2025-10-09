"""
Enhanced LeViT Training Script
============================

Comprehensive training script for the enhanced adaptive LeViT with all advanced features:
- Difficulty-aware training
- Joint policy optimization
- Online learning with coefficient adaptation
- Comprehensive analysis and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
import numpy as np
import time
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from enhanced_adaptive_levit import EnhancedAdaptiveLeViT, create_enhanced_levit, create_fast_enhanced_levit, create_research_enhanced_levit
import levit


def load_datasets(dataset_name='cifar10', batch_size=32, data_path='./data'):
    """Load datasets with appropriate transforms for LeViT (224x224)"""
    
    if dataset_name.lower() == 'cifar10':
        num_classes = 10
        # Enhanced transforms for CIFAR-10 -> 224x224
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to LeViT input size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
        
    elif dataset_name.lower() == 'cifar100':
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, num_classes


def train_static_levit(model_name='LeViT_256', dataset='cifar10', num_epochs=3, batch_size=128, 
                       learning_rate=1e-3, weight_decay=1e-4, warmup_epochs=5, data_path='./data'):
    """Train static LeViT model with identical settings for fair comparison"""
    print(f"\n🔥 Training Static {model_name} for Fair Comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset (same as enhanced)
    train_loader, test_loader, num_classes = load_datasets(dataset, batch_size, data_path)
    
    # Create static model
    model_mapping = {
        'LeViT_128S': levit.LeViT_128S,
        'LeViT_128': levit.LeViT_128,
        'LeViT_192': levit.LeViT_192,
        'LeViT_256': levit.LeViT_256,
        'LeViT_384': levit.LeViT_384
    }
    
    static_model = model_mapping[model_name](
        num_classes=num_classes,
        pretrained=True,
        distillation=False
    ).to(device)
    
    # Same optimizer setup as enhanced
    optimizer = optim.AdamW(
        static_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Same scheduler setup
    warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Static {model_name} - Parameters: {sum(p.numel() for p in static_model.parameters()):,}")
    
    best_accuracy = 0.0
    training_log = {
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epoch_times': []
    }
    
    # Training loop (same structure as enhanced)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        static_model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = static_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(static_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_accuracy += (predicted == labels).float().mean().item()
            
            if batch_idx % 100 == 0:
                print(f"Static Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {running_loss/(batch_idx+1):.4f}, Acc: {running_accuracy/(batch_idx+1):.3f}")
        
        # Update learning rate
        if epoch < warmup_epochs:
            warmup_lr_scheduler.step()
        else:
            main_lr_scheduler.step()
        
        # Evaluation
        static_model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = static_model(images)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)
        test_acc = 100 * test_correct / test_total
        epoch_time = time.time() - epoch_start_time
        
        training_log['train_losses'].append(train_loss)
        training_log['train_accuracies'].append(train_acc)
        training_log['test_accuracies'].append(test_acc)
        training_log['epoch_times'].append(epoch_time)
        
        print(f"📊 Static Epoch {epoch+1}/{num_epochs} Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"   Test Acc: {test_acc:.3f}%, Time: {epoch_time:.1f}s")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        print("-" * 80)
    
    print(f"🎉 Static {model_name} Training Completed!")
    print(f"Best accuracy: {best_accuracy:.3f}")
    
    return static_model, training_log, best_accuracy


def comprehensive_model_comparison(enhanced_model, static_model, test_loader, device, 
                                 enhanced_results, static_accuracy, model_name, dataset):
    """Comprehensive comparison between enhanced and static models"""
    print(f"\n🆚 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    # Set models to eval mode
    enhanced_model.eval()
    static_model.eval()
    enhanced_model.training_mode = False
    
    # GPU warmup
    print("🔥 GPU warmup...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 5:
                break
            images = images.to(device)
            _ = static_model(images)
            _ = enhanced_model(images)
    print("✅ Warmup completed")
    
    # Timing and power comparison
    static_times = []
    enhanced_times = []
    static_correct = 0
    enhanced_correct = 0
    total_samples = 0
    
    enhanced_exit_counts = torch.zeros(4)
    enhanced_computation_costs = []
    
    print("📊 Running comprehensive comparison...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 50:  # Test on 50 batches
                break
                
            images, labels = images.to(device), labels.to(device)
            total_samples += labels.size(0)
            
            # Static model timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            static_outputs = static_model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            static_times.append((time.time() - start_time) * 1000)
            
            _, static_pred = torch.max(static_outputs, 1)
            static_correct += (static_pred == labels).sum().item()
            
            # Enhanced model timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            enhanced_outputs, exit_points, computation_costs = enhanced_model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            enhanced_times.append((time.time() - start_time) * 1000)
            
            _, enhanced_pred = torch.max(enhanced_outputs, 1)
            enhanced_correct += (enhanced_pred == labels).sum().item()
            
            # Track exit statistics
            for exit_point in exit_points:
                enhanced_exit_counts[exit_point.item() - 1] += 1
            enhanced_computation_costs.extend(computation_costs.tolist())
    
    # Calculate results
    static_accuracy_final = 100 * static_correct / total_samples
    enhanced_accuracy_final = 100 * enhanced_correct / total_samples
    static_time = np.mean(static_times)
    enhanced_time = np.mean(enhanced_times)
    actual_speedup = static_time / enhanced_time
    
    exit_distribution = (enhanced_exit_counts / total_samples).tolist()
    avg_comp_cost = np.mean(enhanced_computation_costs)
    theoretical_speedup = 1.0 / avg_comp_cost
    
    # Results
    print(f"\n📊 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n🏛️  STATIC {model_name}:")
    print(f"   Accuracy: {static_accuracy_final:.2f}%")
    print(f"   Inference Time: {static_time:.2f} ms/batch")
    print(f"   Parameters: {sum(p.numel() for p in static_model.parameters()):,}")
    
    print(f"\n🚀 ENHANCED ADAPTIVE {model_name}:")
    print(f"   Accuracy: {enhanced_accuracy_final:.2f}%")
    print(f"   Inference Time: {enhanced_time:.2f} ms/batch")
    print(f"   Parameters: {sum(p.numel() for p in static_model.parameters()):,} + adaptive components")
    print(f"   Exit Distribution: {[f'{x:.1f}%' for x in (np.array(exit_distribution) * 100)]}")
    print(f"   Avg Computation Cost: {avg_comp_cost:.3f}")
    
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    accuracy_diff = static_accuracy_final - enhanced_accuracy_final
    print(f"   Accuracy Difference: {accuracy_diff:.2f}% ({'loss' if accuracy_diff > 0 else 'gain'})")
    print(f"   Actual Speedup: {actual_speedup:.2f}x")
    print(f"   Theoretical Speedup: {theoretical_speedup:.2f}x")
    print(f"   Efficiency: {(actual_speedup/theoretical_speedup)*100:.1f}%")
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if actual_speedup > 1.5 and abs(accuracy_diff) < 3.0:
        print(f"   🎉 EXCELLENT! Enhanced model achieves great speedup with good accuracy")
    elif actual_speedup > 1.2 and abs(accuracy_diff) < 5.0:
        print(f"   ✅ GOOD! Enhanced model shows clear improvement")
    elif actual_speedup > 1.0:
        print(f"   ✅ POSITIVE! Enhanced model is faster")
    else:
        print(f"   ⚠️ Enhanced model needs optimization")
    
    return {
        'static_accuracy': static_accuracy_final,
        'static_time': static_time,
        'enhanced_accuracy': enhanced_accuracy_final,
        'enhanced_time': enhanced_time,
        'actual_speedup': actual_speedup,
        'theoretical_speedup': theoretical_speedup,
        'exit_distribution': exit_distribution,
        'avg_computation_cost': avg_comp_cost
    }


def train_enhanced_levit(
    model_name='LeViT_256',
    dataset='cifar10',
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=5e-2,
    warmup_epochs=5,
    use_all_features=True,
    save_dir='./enhanced_levit_checkpoints',
    log_interval=100,
    run_comparison=True
):
    """
    Train Enhanced Adaptive LeViT with all advanced features
    
    Args:
        model_name: LeViT model variant ('LeViT_128S', 'LeViT_128', 'LeViT_192', 'LeViT_256', 'LeViT_384')
        dataset: Dataset name ('cifar10', 'cifar100')
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs for progressive training
        use_all_features: Whether to enable all advanced features
        save_dir: Directory to save checkpoints and logs
        log_interval: Logging interval in batches
        run_comparison: Whether to train static model and run comparison
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Enhanced Adaptive LeViT on {device}")
    print(f"Model: {model_name}, Dataset: {dataset}, Epochs: {num_epochs}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    train_loader, test_loader, num_classes = load_datasets(dataset, batch_size)
    print(f"Dataset: {dataset} with {num_classes} classes")
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create enhanced model with FIXED aggressive thresholds
    model = create_fast_enhanced_levit(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        use_aggressive_thresholds=True,  # Use aggressive fixed thresholds
        use_progressive_thresholds=False,  # Disable progressive thresholds
        use_difficulty_scaling=use_all_features,
        use_joint_policy=use_all_features,
        use_cost_awareness=use_all_features,
        use_attention_confidence=use_all_features
    )
    model = model.to(device)
    
    print(f"🧠 Model Features:")
    print(f"   - Difficulty Scaling: {model.use_difficulty_scaling}")
    print(f"   - Joint Policy: {model.use_joint_policy}")
    print(f"   - Cost Awareness: {model.use_cost_awareness}")
    print(f"   - Attention Confidence: {model.use_attention_confidence}")
    print(f"   - Number of Exits: {model.n_exits}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    
    # Training logs
    training_log = {
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'exit_distributions': [],
        'computation_costs': [],
        'alpha_statistics': [],
        'online_learning_stats': [],
        'epoch_times': []
    }
    
    best_accuracy = 0.0
    best_model_state = None
    
    print(f"\n🎯 Starting Enhanced Training...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        model.training_mode = True
        
        # Set fixed aggressive thresholds for all epochs
        if hasattr(model, 'use_fixed_thresholds'):
            model.use_fixed_thresholds = True
        if hasattr(model, 'fixed_thresholds'):
            model.fixed_thresholds = [0.1, 0.2, 0.3]
            print(f"🎯 Fixed thresholds for epoch {epoch+1}: {model.fixed_thresholds}")
        
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0
        
        # Progressive training: gradually enable advanced features
        training_progress = epoch / max(num_epochs, 1)
        model.set_training_progress(training_progress)
        
        # Keep consistent exit loss weights for all epochs
        if hasattr(model, 'exit_loss_weights'):
            model.exit_loss_weights = [0.15, 0.25, 0.35, 0.25]
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with advanced training step
            if use_all_features:
                loss = model.train_step(images, labels)
            else:
                # Simple training without advanced features
                outputs = model(images)
                loss, loss_dict = model.compute_training_loss(
                    outputs, labels, epoch, num_epochs
                )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy from final output
            with torch.no_grad():
                if isinstance(model(images), list):
                    final_output = model(images)[-1]  # Final exit output
                else:
                    final_output = model(images)
                _, predicted = torch.max(final_output, 1)
                accuracy = (predicted == labels).float().mean()
                running_accuracy += accuracy.item()
            
            batch_count += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_acc = running_accuracy / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
        
        # Update learning rate
        if epoch < warmup_epochs:
            warmup_lr_scheduler.step()
        else:
            main_lr_scheduler.step()
        
        # Evaluation phase
        model.eval()
        model.training_mode = False
        
        test_results = evaluate_enhanced_levit(model, test_loader, device, use_online_learning=False)
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)
        test_acc = test_results['accuracy']
        
        training_log['train_losses'].append(train_loss)
        training_log['train_accuracies'].append(train_acc)
        training_log['test_accuracies'].append(test_acc)
        training_log['exit_distributions'].append(test_results['exit_distribution'])
        training_log['computation_costs'].append(test_results['avg_computation_cost'])
        training_log['epoch_times'].append(epoch_time)
        
        if use_all_features:
            # Alpha statistics
            if hasattr(model, 'alpha_values') and model.alpha_values:
                alpha_stats = {
                    'mean': np.mean(model.alpha_values[-1000:]),  # Last 1000 samples
                    'std': np.std(model.alpha_values[-1000:]),
                    'min': np.min(model.alpha_values[-1000:]),
                    'max': np.max(model.alpha_values[-1000:])
                }
                training_log['alpha_statistics'].append(alpha_stats)
            
            # Online learning statistics
            online_stats = model.get_online_learning_stats()
            training_log['online_learning_stats'].append(online_stats)
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"   Test Acc: {test_acc:.3f}, Time: {epoch_time:.1f}s")
        print(f"   Exit Distribution: {test_results['exit_distribution']}")
        print(f"   Avg Computation Cost: {test_results['avg_computation_cost']:.3f}")
        print(f"   Theoretical Speedup: {test_results['theoretical_speedup']:.2f}x")
        
        if use_all_features and hasattr(model, 'alpha_values') and model.alpha_values:
            alpha_stats = training_log['alpha_statistics'][-1]
            print(f"   Alpha Stats - Mean: {alpha_stats['mean']:.3f}, Std: {alpha_stats['std']:.3f}")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': test_acc,
                'model_config': {
                    'model_name': model_name,
                    'num_classes': num_classes,
                    'use_all_features': use_all_features
                }
            }
            
            # Save best model checkpoint
            best_model_path = os.path.join(save_dir, f'best_{model_name}_{dataset}.pth')
            torch.save(best_model_state, best_model_path)
            
            # Save adaptive state
            if use_all_features:
                adaptive_state_path = os.path.join(save_dir, f'adaptive_state_{model_name}_{dataset}.json')
                model.save_adaptive_state(adaptive_state_path)
            
            print(f"   💾 New best model saved! Accuracy: {test_acc:.3f}")
        
        # Periodic misclassification analysis
        if use_all_features and (epoch + 1) % 10 == 0:
            print(f"\n🔍 Performing misclassification analysis...")
            misclass_results = model.analyze_misclassifications_with_class_specificity(
                test_loader, adjustment_factor=0.05
            )
            
            # Log misclassification insights
            problematic_classes = []
            for class_id, analysis in misclass_results['class_analysis'].items():
                if analysis['problematic_exits']:
                    problematic_classes.append(class_id)
            
            if problematic_classes:
                print(f"   ⚠️  Problematic classes detected: {problematic_classes}")
                print(f"   🔧 Adaptive coefficients updated for improved performance")
        
        print("-" * 80)
    
    # Final evaluation and analysis
    print(f"\n🎉 Training completed!")
    print(f"Best accuracy: {best_accuracy:.3f}")
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # Run comprehensive comparison if requested
    comparison_results = None
    if run_comparison:
        print(f"\n📋 Training Static Model for Fair Comparison...")
        static_model, static_log, static_best_acc = train_static_levit(
            model_name=model_name,
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=1e-4,  # Use lighter weight decay for static model
            warmup_epochs=warmup_epochs,
            data_path='./data'
        )
        
        # Set aggressive thresholds for enhanced model during comparison
        if hasattr(model, 'use_fixed_thresholds'):
            model.use_fixed_thresholds = True
        if hasattr(model, 'fixed_thresholds'):
            model.fixed_thresholds = [0.1, 0.2, 0.3]
        print(f"🎯 Enhanced model thresholds set to: {model.fixed_thresholds}")
        
        print(f"\n📋 Running Comprehensive Model Comparison...")
        comparison_results = comprehensive_model_comparison(
            model, static_model, test_loader, device, None, 
            static_best_acc, model_name, dataset
        )
        
        # Save comparison results
        comparison_path = os.path.join(save_dir, f'comparison_results_{model_name}_{dataset}.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"💾 Comparison results saved to: {comparison_path}")
    else:
        # Original comprehensive evaluation
        print(f"\n📋 Final Comprehensive Evaluation...")
        final_results = comprehensive_evaluation(model, test_loader, device)
    
    # Save training log
    log_path = os.path.join(save_dir, f'training_log_{model_name}_{dataset}.json')
    with open(log_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_log = {}
        for key, value in training_log.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_log[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                serializable_log[key] = value
        json.dump(serializable_log, f, indent=2)
    
    # Generate training plots
    generate_training_plots(training_log, save_dir, model_name, dataset)
    
    return model, training_log, None, comparison_results


def evaluate_enhanced_levit(model, test_loader, device, use_online_learning=True):
    """
    Comprehensive evaluation of Enhanced Adaptive LeViT
    """
    model.eval()
    model.training_mode = False
    
    total_correct = 0
    total_samples = 0
    exit_counts = torch.zeros(model.n_exits)
    total_computation_cost = 0.0
    
    inference_times = []
    alpha_values_collected = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Clear previous analysis data
            model.clear_analysis_data(preserve_confidence_stats=True)
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Run inference with online learning if enabled
            if use_online_learning:
                outputs, exit_points, computation_costs = model.inference_with_online_learning(images, labels)
            else:
                outputs, exit_points, computation_costs = model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Track exit distributions
            for exit_point in exit_points:
                exit_counts[exit_point.item() - 1] += 1
            
            # Track computation costs
            total_computation_cost += computation_costs.sum().item()
            
            # Collect alpha values if available
            if hasattr(model, 'alpha_values') and model.alpha_values:
                alpha_values_collected.extend(model.alpha_values[-labels.size(0):])
    
    # Calculate metrics
    accuracy = 100 * total_correct / total_samples
    exit_distribution = (exit_counts / total_samples).tolist()
    avg_computation_cost = total_computation_cost / total_samples
    theoretical_speedup = 1.0 / avg_computation_cost
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    results = {
        'accuracy': accuracy,
        'exit_distribution': exit_distribution,
        'exit_counts': exit_counts.tolist(),
        'avg_computation_cost': avg_computation_cost,
        'theoretical_speedup': theoretical_speedup,
        'avg_inference_time': avg_inference_time,
        'total_samples': total_samples
    }
    
    if alpha_values_collected:
        results['alpha_statistics'] = {
            'mean': np.mean(alpha_values_collected),
            'std': np.std(alpha_values_collected),
            'min': np.min(alpha_values_collected),
            'max': np.max(alpha_values_collected)
        }
    
    return results


def comprehensive_evaluation(model, test_loader, device):
    """
    Perform comprehensive evaluation with detailed analysis
    """
    print("🔬 Running comprehensive evaluation...")
    
    # Standard evaluation
    standard_results = evaluate_enhanced_levit(model, test_loader, device, use_online_learning=True)
    
    # Threshold sensitivity analysis
    print("🎯 Analyzing threshold sensitivity...")
    threshold_results = analyze_threshold_sensitivity(model, test_loader, device)
    
    # Online learning performance
    if hasattr(model, 'adaptive_coeff_manager'):
        print("📚 Analyzing online learning performance...")
        online_learning_stats = model.get_online_learning_stats()
    else:
        online_learning_stats = {}
    
    # Cost-benefit analysis
    print("💰 Performing cost-benefit analysis...")
    cost_benefit = analyze_cost_benefit(model, test_loader, device)
    
    comprehensive_results = {
        'standard_evaluation': standard_results,
        'threshold_sensitivity': threshold_results,
        'online_learning_stats': online_learning_stats,
        'cost_benefit_analysis': cost_benefit
    }
    
    return comprehensive_results


def analyze_threshold_sensitivity(model, test_loader, device):
    """
    Analyze model performance across different threshold settings
    """
    original_thresholds = model.exit_thresholds.copy()
    threshold_configs = [
        [0.8, 0.75, 0.7],   # Conservative
        [0.75, 0.7, 0.6],   # Default
        [0.7, 0.6, 0.5],    # Aggressive
        [0.6, 0.5, 0.4]     # Very aggressive
    ]
    
    results = []
    
    for thresholds in threshold_configs:
        model.exit_thresholds = thresholds
        eval_results = evaluate_enhanced_levit(model, test_loader, device, use_online_learning=False)
        
        results.append({
            'thresholds': thresholds,
            'accuracy': eval_results['accuracy'],
            'speedup': eval_results['theoretical_speedup'],
            'exit_distribution': eval_results['exit_distribution']
        })
    
    # Restore original thresholds
    model.exit_thresholds = original_thresholds
    
    return results


def analyze_cost_benefit(model, test_loader, device):
    """
    Analyze cost-benefit trade-offs of early exit decisions
    """
    model.eval()
    model.training_mode = False
    
    total_samples = 0
    accuracy_by_exit = [0] * model.n_exits
    samples_by_exit = [0] * model.n_exits
    computation_by_exit = [0] * model.n_exits
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs, exit_points, computation_costs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels)
            
            for i in range(len(exit_points)):
                exit_idx = exit_points[i].item() - 1
                accuracy_by_exit[exit_idx] += correct[i].item()
                samples_by_exit[exit_idx] += 1
                computation_by_exit[exit_idx] += computation_costs[i].item()
            
            total_samples += len(exit_points)
    
    # Calculate metrics
    cost_benefit = {}
    for exit_idx in range(model.n_exits):
        if samples_by_exit[exit_idx] > 0:
            cost_benefit[f'exit_{exit_idx + 1}'] = {
                'accuracy': accuracy_by_exit[exit_idx] / samples_by_exit[exit_idx],
                'usage_percentage': samples_by_exit[exit_idx] / total_samples * 100,
                'avg_computation_cost': computation_by_exit[exit_idx] / samples_by_exit[exit_idx],
                'sample_count': samples_by_exit[exit_idx]
            }
    
    return cost_benefit


def generate_training_plots(training_log, save_dir, model_name, dataset):
    """
    Generate comprehensive training plots
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # Use default style
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Enhanced {model_name} Training Analysis - {dataset.upper()}', fontsize=16)
    
    epochs = range(1, len(training_log['train_losses']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, training_log['train_losses'], 'b-', label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, training_log['train_accuracies'], 'g-', label='Train Acc')
    axes[0, 1].plot(epochs, training_log['test_accuracies'], 'r-', label='Test Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Computation cost plot
    axes[0, 2].plot(epochs, training_log['computation_costs'], 'm-', label='Avg Computation Cost')
    axes[0, 2].set_title('Computation Cost Over Time')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Computation Cost')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Exit distribution evolution
    if training_log['exit_distributions']:
        exit_data = np.array(training_log['exit_distributions'])
        for i in range(exit_data.shape[1]):
            axes[1, 0].plot(epochs, exit_data[:, i], label=f'Exit {i+1}')
        axes[1, 0].set_title('Exit Distribution Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Alpha statistics
    if training_log['alpha_statistics']:
        alpha_means = [stats['mean'] for stats in training_log['alpha_statistics']]
        alpha_stds = [stats['std'] for stats in training_log['alpha_statistics']]
        alpha_epochs = range(1, len(alpha_means) + 1)
        
        axes[1, 1].plot(alpha_epochs, alpha_means, 'b-', label='Mean α')
        axes[1, 1].fill_between(alpha_epochs, 
                               np.array(alpha_means) - np.array(alpha_stds),
                               np.array(alpha_means) + np.array(alpha_stds),
                               alpha=0.3)
        axes[1, 1].set_title('Difficulty Score (α) Statistics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('α Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Training time per epoch
    axes[1, 2].plot(epochs, training_log['epoch_times'], 'orange', label='Epoch Time')
    axes[1, 2].set_title('Training Time per Epoch')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_analysis_{model_name}_{dataset}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to: {plot_path}")


def benchmark_enhanced_vs_static(model_name='LeViT_256', dataset='cifar10', batch_size=32):
    """
    Benchmark enhanced adaptive LeViT against static LeViT
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    _, test_loader, num_classes = load_datasets(dataset, batch_size)
    
    # Create models
    from levit import LeViT_256, LeViT_384, LeViT_128, LeViT_192, LeViT_128S
    model_mapping = {
        'LeViT_128S': LeViT_128S,
        'LeViT_128': LeViT_128,
        'LeViT_192': LeViT_192,
        'LeViT_256': LeViT_256,
        'LeViT_384': LeViT_384
    }
    
    static_model = model_mapping[model_name](
        num_classes=num_classes, pretrained=False, distillation=False
    ).to(device)
    
    enhanced_model = create_enhanced_levit(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        use_difficulty_scaling=True,
        use_joint_policy=True,
        use_cost_awareness=True,
        use_attention_confidence=True
    ).to(device)
    
    # Set to eval mode
    static_model.eval()
    enhanced_model.eval()
    enhanced_model.training_mode = False
    
    print(f"🏁 Benchmarking {model_name} on {dataset.upper()}")
    print(f"Device: {device}")
    
    # Benchmark static model
    static_times = []
    static_correct = 0
    static_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            outputs = static_model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            static_times.append(time.time() - start_time)
            
            _, predicted = torch.max(outputs, 1)
            static_correct += (predicted == labels).sum().item()
            static_total += labels.size(0)
    
    # Benchmark enhanced model
    enhanced_results = evaluate_enhanced_levit(enhanced_model, test_loader, device)
    
    # Calculate results
    static_accuracy = 100 * static_correct / static_total
    static_time = np.mean(static_times) * 1000  # ms
    
    enhanced_accuracy = enhanced_results['accuracy']
    enhanced_time = enhanced_results['avg_inference_time']
    speedup = static_time / enhanced_time
    
    print(f"\n📊 Benchmark Results:")
    print(f"Static {model_name}:")
    print(f"   Accuracy: {static_accuracy:.2f}%")
    print(f"   Inference Time: {static_time:.2f} ms/batch")
    
    print(f"\nEnhanced Adaptive {model_name}:")
    print(f"   Accuracy: {enhanced_accuracy:.2f}%")
    print(f"   Inference Time: {enhanced_time:.2f} ms/batch")
    print(f"   Theoretical Speedup: {enhanced_results['theoretical_speedup']:.2f}x")
    print(f"   Actual Speedup: {speedup:.2f}x")
    print(f"   Exit Distribution: {enhanced_results['exit_distribution']}")
    
    accuracy_drop = static_accuracy - enhanced_accuracy
    print(f"\n💡 Analysis:")
    print(f"   Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"   Speed Gain: {speedup:.2f}x")
    print(f"   Efficiency Score: {speedup / max(1, accuracy_drop):.2f}")
    
    if speedup > 1.5 and accuracy_drop < 2.0:
        print(f"   🎉 EXCELLENT trade-off achieved!")
    elif speedup > 1.2 and accuracy_drop < 3.0:
        print(f"   ✅ Good trade-off achieved!")
    else:
        print(f"   ⚠️  Consider tuning thresholds for better trade-off")
    
    return {
        'static_accuracy': static_accuracy,
        'static_time': static_time,
        'enhanced_accuracy': enhanced_accuracy,
        'enhanced_time': enhanced_time,
        'speedup': speedup,
        'theoretical_speedup': enhanced_results['theoretical_speedup'],
        'exit_distribution': enhanced_results['exit_distribution']
    }


if __name__ == "__main__":
    # Example training run
    model, training_log, final_results, comparison_results = train_enhanced_levit(
        model_name='LeViT_256',
        dataset='cifar10',
        num_epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        use_all_features=True,
        run_comparison=True
    )
    
    print(f"\n🎯 Training completed successfully!")
    if final_results:
        print(f"Final results: {final_results['standard_evaluation']}")
    if comparison_results:
        print(f"\n🆚 COMPARISON SUMMARY:")
        print(f"   Enhanced Accuracy: {comparison_results['enhanced_accuracy']:.2f}%")
        print(f"   Static Accuracy: {comparison_results['static_accuracy']:.2f}%")
        print(f"   Actual Speedup: {comparison_results['actual_speedup']:.2f}x")
        print(f"   Theoretical Speedup: {comparison_results['theoretical_speedup']:.2f}x")
        print(f"   Exit Distribution: {[f'{x:.1f}%' for x in (np.array(comparison_results['exit_distribution']) * 100)]}")