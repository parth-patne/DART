import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import pickle

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

import pynvml
import pandas as pd
import threading
import queue

from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns


class ThresholdQLearningAgent:
   
    def __init__(self, n_exits=5, alpha_bins=10, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.n_exits = n_exits
        self.alpha_bins = alpha_bins
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Q-table: state -> [coeff_decrease, coeff_maintain, coeff_increase]
        self.q_table = defaultdict(lambda: np.zeros(3))
        
        # Coefficient adjustment amounts
        self.coeff_adjustments = [-0.1, 0.0, 0.1]
        
    def get_state(self, alpha_value, exit_idx):
        
        alpha_bin = min(int(alpha_value * self.alpha_bins), self.alpha_bins - 1)
        return (alpha_bin, exit_idx)
    
    def select_action(self, state, training=True):
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def get_coefficient_adjustment(self, alpha_value, exit_idx, training=True):
        
        state = self.get_state(alpha_value, exit_idx)
        action = self.select_action(state, training)
        return self.coeff_adjustments[action], action, state
    
    def export_q_table(self):
        
        return {k: v.copy() for k, v in self.q_table.items()}


class QLearningAgent:
    @staticmethod
    def _q_table_factory():
        return np.zeros(2)

    def __init__(self, n_exits, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.n_exits = n_exits
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(QLearningAgent._q_table_factory)

    def export_q_table(self):
        return {k: v.copy() for k, v in self.q_table.items()}

    def get_state(self, layer_idx, confidence):
        conf_bin = int(confidence * 10)
        return (layer_idx, conf_bin)

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


class DifficultyEstimator(nn.Module):
    
    def __init__(self, w1=0.4, w2=0.3, w3=0.3):
        super(DifficultyEstimator, self).__init__()
        self.w1 = w1  
        self.w2 = w2  
        self.w3 = w3  

        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def compute_edge_density(self, x):
        
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = x

        
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        
        edge_density = grad_magnitude.mean(dim=[1, 2, 3])
        edge_density = torch.clamp(edge_density, 0, 1)

        return edge_density

    def compute_pixel_variance(self, x):
        
        
        variance = torch.var(x, dim=[2, 3])  # [B, C]

        
        variance = variance.mean(dim=1)  # [B]
        variance = torch.clamp(variance, 0, 1)

        return variance

    def compute_gradient_complexity(self, x):
        
        
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                dtype=torch.float32, device=x.device)
        laplacian = laplacian.unsqueeze(0).unsqueeze(0)

        
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = x

        
        laplacian_response = F.conv2d(gray, laplacian, padding=1)

        
        complexity = torch.abs(laplacian_response).mean(dim=[1, 2, 3])
        complexity = torch.clamp(complexity, 0, 1)

        return complexity

    def forward(self, x):
        
        edge_density = self.compute_edge_density(x)
        pixel_variance = self.compute_pixel_variance(x)
        gradient_complexity = self.compute_gradient_complexity(x)

        
        alpha = (self.w1 * edge_density +
                self.w2 * pixel_variance +
                self.w3 * gradient_complexity)

        
        alpha = torch.clamp(alpha, 0, 1)

        return alpha, {
            'edge_density': edge_density,
            'pixel_variance': pixel_variance,
            'gradient_complexity': gradient_complexity
        }


class JointExitPolicy:
    
    def __init__(self, n_exits=5, alpha_bins=20, confidence_bins=20,
                 gamma=0.95, convergence_threshold=1e-6):
        self.n_exits = n_exits
        self.alpha_bins = alpha_bins
        self.confidence_bins = confidence_bins
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold

        
        self.value_table = np.zeros((n_exits, alpha_bins, confidence_bins))
        self.policy_table = np.zeros((n_exits, alpha_bins, confidence_bins), dtype=int)

        
        self.computation_costs = np.array([0.2, 0.45, 0.7, 0.9, 1.0])
        self.accuracy_rewards = np.array([0.75, 0.85, 0.9, 0.93, 1.0])

        
        self.learning_rate = 0.1
        self.is_trained = False

    def discretize_state(self, exit_idx, alpha, confidence):
        
        alpha_bin = min(int(alpha * self.alpha_bins), self.alpha_bins - 1)
        conf_bin = min(int(confidence * self.confidence_bins), self.confidence_bins - 1)
        return exit_idx, alpha_bin, conf_bin

    def get_reward(self, exit_idx, action, correct_prediction, alpha):
        
        if action == 1:  # Exit
            accuracy_reward = 10.0 if correct_prediction else -10.0
            efficiency_bonus = (self.n_exits - exit_idx) * 2.0
            difficulty_adjustment = (1 - alpha) * 1.0
            return accuracy_reward + efficiency_bonus + difficulty_adjustment
        else:  
            continuation_cost = -self.computation_costs[exit_idx] * 0.5
            return continuation_cost

    def value_iteration(self, max_iterations=1000):
        
        print("Performing value iteration for joint exit policy...")

        for iteration in range(max_iterations):
            old_values = self.value_table.copy()

            for exit_idx in range(self.n_exits):
                for alpha_bin in range(self.alpha_bins):
                    for conf_bin in range(self.confidence_bins):
                        alpha = alpha_bin / self.alpha_bins
                        confidence = conf_bin / self.confidence_bins

                        
                        q_continue = 0
                        q_exit = 0

                        if exit_idx < self.n_exits - 1:
                            continuation_reward = -self.computation_costs[exit_idx] * 0.5
                            future_alpha_bin = min(alpha_bin + 1, self.alpha_bins - 1)
                            future_conf_bin = min(conf_bin + 1, self.confidence_bins - 1)
                            future_value = self.value_table[exit_idx + 1, future_alpha_bin, future_conf_bin]
                            q_continue = continuation_reward + self.gamma * future_value

                        expected_accuracy = self.accuracy_rewards[exit_idx]
                        exit_reward = expected_accuracy * 10.0 + (self.n_exits - exit_idx) * 2.0
                        q_exit = exit_reward

                        if q_exit > q_continue:
                            self.value_table[exit_idx, alpha_bin, conf_bin] = q_exit
                            self.policy_table[exit_idx, alpha_bin, conf_bin] = 1
                        else:
                            self.value_table[exit_idx, alpha_bin, conf_bin] = q_continue
                            self.policy_table[exit_idx, alpha_bin, conf_bin] = 0

            if np.max(np.abs(self.value_table - old_values)) < self.convergence_threshold:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break

        self.is_trained = True
        print("Joint exit policy training completed")

    def get_action(self, exit_idx, alpha, confidence):
        
        if not self.is_trained:
            if exit_idx >= self.n_exits - 1:
                return 1
            dynamic_threshold = 0.7 - 0.4 * alpha + 0.1 * exit_idx / self.n_exits
            return 1 if confidence > dynamic_threshold else 0

        exit_idx = min(exit_idx, self.n_exits - 1)
        _, alpha_bin, conf_bin = self.discretize_state(exit_idx, alpha, confidence)
        return self.policy_table[exit_idx, alpha_bin, conf_bin]

    def update_online(self, exit_idx, alpha, confidence, action, reward, next_alpha=None, next_confidence=None):
        
        state = self.discretize_state(exit_idx, alpha, confidence)

        if action == 0 and exit_idx < self.n_exits - 1 and next_alpha is not None and next_confidence is not None:
            next_state = self.discretize_state(exit_idx + 1, next_alpha, next_confidence)
            next_value = self.value_table[next_state]
            target = reward + self.gamma * next_value
        else:
            target = reward

        current_value = self.value_table[state]
        td_error = target - current_value
        self.value_table[state] += self.learning_rate * td_error

        if exit_idx < self.n_exits - 1:
            q_continue = self.value_table[state] if action == 0 else current_value
            q_exit = self.value_table[state] if action == 1 else current_value
            self.policy_table[state] = 1 if q_exit > q_continue else 0


class AdaptiveCoefficientManager:
    
    def __init__(self, n_exits=5, strategy='adaptive_decay', initial_coeffs=None):
        self.n_exits = n_exits
        self.strategy = strategy

        if initial_coeffs is not None:
            self.base_coeffs = np.array(initial_coeffs)
        else:
            self.base_coeffs = np.linspace(1.2, 0.7, n_exits)

        self.class_sensitivity = np.ones(10)
        self.layer_sensitivity = np.linspace(0.8, 1.2, n_exits)
        self.temporal_decay = 0.95
        self.adaptation_history = []
        self.class_specific_coefficients = {class_id: list(self.base_coeffs) for class_id in range(10)}

        self.online_learning_enabled = True
        self.learning_rate = 0.01
        self.performance_window = 100
        self.recent_performance = []
        self.class_performance_tracking = {class_id: [] for class_id in range(10)}
        self.coefficient_update_frequency = 50

    def get_coefficients(self, exit_idx, alpha_val, class_id=None,
                        sample_count=0, accuracy_history=None):
        
        base_coeff = self.base_coeffs[exit_idx]
        coeff = base_coeff # Default
        if self.strategy == 'adaptive_decay':
            depth_factor = np.exp(-0.3 * exit_idx)
            difficulty_factor = 1.0 + 0.5 * alpha_val
            coeff = base_coeff * depth_factor * difficulty_factor
        # Add other strategies if needed...
        return np.clip(coeff, 0.3, 2.5)

    def update_coefficient(self, exit_idx, confidence, predicted_class, alpha_score, should_exit=None):
        
        if (hasattr(self, 'class_specific_coefficients') and
            predicted_class in self.class_specific_coefficients):
            base_coeff = self.class_specific_coefficients[predicted_class][exit_idx]
            if self.strategy == 'class_aware':
                return base_coeff * (1.0 + 0.2 * alpha_score)
            else:
                return base_coeff
        else:
            return self.get_coefficients(exit_idx, alpha_score, predicted_class)

    def online_learning_update(self, exit_idx, alpha_score, predicted_class, was_correct, confidence):
        
        if not self.online_learning_enabled:
            return

        self.class_performance_tracking[predicted_class].append({
            'exit_idx': exit_idx, 'correct': was_correct,
            'confidence': confidence, 'alpha_score': alpha_score
        })

        if len(self.class_performance_tracking[predicted_class]) > self.performance_window:
            self.class_performance_tracking[predicted_class] = \
                self.class_performance_tracking[predicted_class][-self.performance_window:]

        if len(self.class_performance_tracking[predicted_class]) >= self.coefficient_update_frequency:
            self._update_class_coefficients(predicted_class)

    def _update_class_coefficients(self, class_id):
        
        performance_data = self.class_performance_tracking[class_id]
        exit_performance = {}
        for exit_idx in range(self.n_exits):
            exit_samples = [p for p in performance_data if p['exit_idx'] == exit_idx]
            if len(exit_samples) > 5:
                accuracy = sum(1 for p in exit_samples if p['correct']) / len(exit_samples)
                exit_performance[exit_idx] = {'accuracy': accuracy}

        for exit_idx, perf in exit_performance.items():
            current_coeff = self.class_specific_coefficients[class_id][exit_idx]
            target_accuracy = 0.85
            if perf['accuracy'] < target_accuracy:
                adjustment = self.learning_rate * (target_accuracy - perf['accuracy'])
                new_coeff = current_coeff + adjustment
            else:
                adjustment = self.learning_rate * (perf['accuracy'] - target_accuracy) * 0.5
                new_coeff = current_coeff - adjustment
            self.class_specific_coefficients[class_id][exit_idx] = np.clip(new_coeff, 0.1, 2.0)

    def save_online_learning_state(self, filepath):
        
        state = {
            'class_specific_coefficients': {k: v for k, v in self.class_specific_coefficients.items()},
            'class_performance_tracking': self.class_performance_tracking,
            'learning_rate': self.learning_rate,
            'base_coeffs': self.base_coeffs.tolist(),
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_online_learning_state(self, filepath):
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.class_specific_coefficients = {int(k): v for k, v in state['class_specific_coefficients'].items()}
            self.class_performance_tracking = {int(k): v for k, v in state['class_performance_tracking'].items()}
            self.learning_rate = state['learning_rate']
            self.base_coeffs = np.array(state['base_coeffs'])
            return True
        except Exception as e:
            print(f"Warning: Could not load online learning state: {e}")
            return False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class EarlyExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes, feature_size=4):
        super(EarlyExitBlock, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * feature_size * feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)


class BranchyResNet18(nn.Module):
   
    def __init__(self, num_classes=10, in_channels=3, exit_threshold=0.5):
        super(BranchyResNet18, self).__init__()
        self.num_classes = num_classes
        self.exit_threshold = exit_threshold
        self.training_mode = True
        
        # ResNet-18 Backbone
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Branch 1: After layer1 (early exit)
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Branch 2: After layer2 (middle exit)
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Branch 3: After layer3 (late exit)
        self.branch3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Final classifier (after layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_classifier = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def compute_entropy(self, logits):
        
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=logits.device))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def forward(self, x):
        if self.training_mode:
            return self._forward_training(x)
        else:
            return self._forward_inference(x)
    
    def _forward_training(self, x):
        
        outputs = []
        
        # Forward through initial layers
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Forward through layer1 -> Branch 1
        x = self.layer1(x)
        branch1_out = self.branch1(x)
        outputs.append(branch1_out)
        
        # Forward through layer2 -> Branch 2
        x = self.layer2(x)
        branch2_out = self.branch2(x)
        outputs.append(branch2_out)
        
        # Forward through layer3 -> Branch 3
        x = self.layer3(x)
        branch3_out = self.branch3(x)
        outputs.append(branch3_out)
        
        # Forward through layer4 -> Final classifier
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_out = self.final_classifier(x)
        outputs.append(final_out)
        
        return outputs
    
    def _forward_inference(self, x):
        
        batch_size = x.size(0)
        device = x.device
        
        
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        remaining_indices = torch.arange(batch_size, device=device)
        
        x_current = x
        
        
        x_current = self.relu(self.bn1(self.conv1(x_current)))
        
        
        x_current = self.layer1(x_current)
        branch1_out = self.branch1(x_current)
        
        
        entropy = self.compute_entropy(branch1_out)
        probs = torch.softmax(branch1_out, dim=1)
        max_confidence, _ = torch.max(probs, dim=1)
        
        
        exit_mask = (entropy < self.exit_threshold) & (max_confidence > 0.8)
        
        if exit_mask.any():
            exit_indices = remaining_indices[exit_mask]
            final_outputs[exit_indices] = branch1_out[exit_mask]
            exit_points[exit_indices] = 1
            
            remaining_indices = remaining_indices[~exit_mask]
            x_current = x_current[~exit_mask]
        

        if len(remaining_indices) > 0:
            
            x_current = self.layer2(x_current)
            branch2_out = self.branch2(x_current)
            
            entropy = self.compute_entropy(branch2_out)
            probs = torch.softmax(branch2_out, dim=1)
            max_confidence, _ = torch.max(probs, dim=1)
            
            exit_mask = (entropy < self.exit_threshold * 1.1) & (max_confidence > 0.75)
            
            if exit_mask.any():
                exit_indices = remaining_indices[exit_mask]
                final_outputs[exit_indices] = branch2_out[exit_mask]
                exit_points[exit_indices] = 2
                
                remaining_indices = remaining_indices[~exit_mask]
                x_current = x_current[~exit_mask]
        
        
        if len(remaining_indices) > 0:
            
            x_current = self.layer3(x_current)
            branch3_out = self.branch3(x_current)
            
            entropy = self.compute_entropy(branch3_out)
            probs = torch.softmax(branch3_out, dim=1)
            max_confidence, _ = torch.max(probs, dim=1)
            
            exit_mask = (entropy < self.exit_threshold * 1.2) & (max_confidence > 0.7)
            
            if exit_mask.any():
                exit_indices = remaining_indices[exit_mask]
                final_outputs[exit_indices] = branch3_out[exit_mask]
                exit_points[exit_indices] = 3
                
                remaining_indices = remaining_indices[~exit_mask]
                x_current = x_current[~exit_mask]
        
        
        if len(remaining_indices) > 0:
            x_current = self.layer4(x_current)
            x_current = self.avgpool(x_current)
            x_current = torch.flatten(x_current, 1)
            final_out = self.final_classifier(x_current)
            final_outputs[remaining_indices] = final_out
            exit_points[remaining_indices] = 4
        
        return final_outputs, exit_points


class StaticResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(StaticResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FullLayerResNet18(nn.Module):
    
    def __init__(self, num_classes=10, in_channels=3, use_difficulty_scaling=True,
                 use_joint_policy=True, use_cost_awareness=True):
        super(FullLayerResNet18, self).__init__()
        self.num_classes = num_classes
        self.training_mode = True
        self.n_exits = 5  # After layer1, layer2, layer3, layer4, final

        self.exit_loss_weights = [0.2, 0.3, 0.5, 0.7, 1.0]

        self.use_difficulty_scaling = use_difficulty_scaling
        self.difficulty_estimator = DifficultyEstimator() if use_difficulty_scaling else None

        self.use_joint_policy = use_joint_policy
        self.joint_policy = JointExitPolicy(n_exits=self.n_exits) if use_joint_policy else None

        self.use_cost_awareness = use_cost_awareness
        self.computation_costs = [0.2, 0.45, 0.7, 0.9, 1.0] # Relative costs for 5 exits
        self.energy_costs = [0.15, 0.4, 0.65, 0.85, 1.0] # Energy costs for 5 exits

        self.alpha_values = []
        self.exit_decisions_log = []
        self.policy_decisions_log = []
        self.cost_analysis_log = []

        self.adaptive_coeff_manager = AdaptiveCoefficientManager(n_exits=self.n_exits)

        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.exit1 = EarlyExitBlock(64, num_classes, feature_size=8)

        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.exit2 = EarlyExitBlock(128, num_classes, feature_size=4)

        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.exit3 = EarlyExitBlock(256, num_classes, feature_size=2)

        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.exit4 = EarlyExitBlock(512, num_classes, feature_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

        if self.use_joint_policy and self.joint_policy:
            self.joint_policy.value_iteration()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.training_mode:
            return self._forward_training(x)
        else:
            return self._forward_inference(x)

    def _forward_training(self, x):
        
        outputs = []
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x); outputs.append(self.exit1(x))
        x = self.layer2(x); outputs.append(self.exit2(x))
        x = self.layer3(x); outputs.append(self.exit3(x))
        x = self.layer4(x); outputs.append(self.exit4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs.append(self.final_classifier(x))

        return outputs

    def _forward_inference(self, x):
        
        device = x.device
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        computation_costs = torch.zeros(batch_size, device=device)
        remaining_indices = torch.arange(batch_size, device=device)

        alpha_scores = None
        if self.use_difficulty_scaling and self.difficulty_estimator is not None:
            alpha_scores, _ = self.difficulty_estimator(x)
            self.alpha_values.extend(alpha_scores.detach().cpu().numpy().tolist())

        x_current = self.relu(self.bn1(self.conv1(x)))

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        exits = [self.exit1, self.exit2, self.exit3, self.exit4]

        for exit_idx, (layer, exit_block) in enumerate(zip(layers, exits)):
            if len(remaining_indices) == 0:
                break

            x_current = layer(x_current)
            exit_output = exit_block(x_current)
            confidence, _ = torch.max(torch.softmax(exit_output, dim=1), dim=1)
            
            exit_decisions = []
            current_alphas = alpha_scores[remaining_indices] if alpha_scores is not None else torch.full((len(remaining_indices),), 0.5, device=device)
            
            for i, rem_idx in enumerate(remaining_indices):
                alpha_val = current_alphas[i].item()
                should_exit = False
                
                if self.use_joint_policy and self.joint_policy and self.joint_policy.is_trained:
                    action = self.joint_policy.get_action(exit_idx, alpha_val, confidence[i].item())
                    should_exit = (action == 1)
                else:
                    adaptive_coeff = self.adaptive_coeff_manager.get_coefficients(exit_idx, alpha_val)
                    should_exit = self.combined_confidence_decision(exit_output[i:i+1], alpha_val, exit_idx, adaptive_coeff=adaptive_coeff).item()
                
                exit_decisions.append(should_exit)

                if not self.training_mode:
                    policy_info = {
                        'exit_idx': exit_idx,
                        'alpha': alpha_val,
                        'confidence': confidence[i].item(),
                       
                        'decision': bool(should_exit)
                    }
                    self.policy_decisions_log.append(policy_info)

            exit_mask = torch.tensor(exit_decisions, dtype=torch.bool, device=device)
            exit_indices = remaining_indices[exit_mask]

            if len(exit_indices) > 0:
                final_outputs[exit_indices] = exit_output[exit_mask]
                exit_points[exit_indices] = exit_idx + 1
                computation_costs[exit_indices] = self.computation_costs[exit_idx]
                if not self.training_mode:
                     self.exit_decisions_log.append({
                        'sample_indices': exit_indices.cpu().numpy().tolist(),
                        'exit_point': exit_idx + 1,
                        'alpha_values': alpha_scores[exit_indices].detach().cpu().numpy().tolist() if alpha_scores is not None else [],
                        'computation_cost': self.computation_costs[exit_idx]
                    })

            remaining_indices = remaining_indices[~exit_mask]
            x_current = x_current[~exit_mask]

        if len(remaining_indices) > 0:
            x_current = self.avgpool(x_current)
            x_current = torch.flatten(x_current, 1)
            final_output = self.final_classifier(x_current)
            final_outputs[remaining_indices] = final_output
            exit_points[remaining_indices] = self.n_exits
            computation_costs[remaining_indices] = self.computation_costs[-1]

        return final_outputs, exit_points, computation_costs

    def train_step(self, x, labels):
        device = x.device
        outputs = self._forward_training(x)
        total_loss = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for output, weight in zip(outputs, self.exit_loss_weights):
            total_loss += weight * criterion(output, labels)

        if self.use_joint_policy and self.joint_policy and self.difficulty_estimator:
            alpha_scores, _ = self.difficulty_estimator(x)
            x_current = self.relu(self.bn1(self.conv1(x)))
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            exits = [self.exit1, self.exit2, self.exit3, self.exit4]
            for exit_idx, (layer, exit_block) in enumerate(zip(layers, exits)):
                x_current = layer(x_current)
                exit_output = exit_block(x_current)
                softmax_out = torch.softmax(exit_output, dim=1)
                confidence, predictions = torch.max(softmax_out, dim=1)
                for i in range(x.size(0)):
                    alpha_val, conf_val = alpha_scores[i].item(), confidence[i].item()
                    correct = (predictions[i] == labels[i]).item()
                    action = self.joint_policy.get_action(exit_idx, alpha_val, conf_val)
                    reward = self.joint_policy.get_reward(exit_idx, action, correct, alpha_val)
                    if exit_idx < self.n_exits - 2:
                        self.joint_policy.update_online(exit_idx, alpha_val, conf_val, action, reward, alpha_val, conf_val) # Simplified
                    else:
                        self.joint_policy.update_online(exit_idx, alpha_val, conf_val, action, reward)
        return total_loss

    def compute_entropy_confidence(self, logits, method='normalized_entropy', temperature=1.0):
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        if method == 'normalized_entropy':
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=logits.device))
            confidence = 1.0 - (entropy / max_entropy)
        else: # max_prob
            confidence, _ = torch.max(probs, dim=1)
        return confidence

    def combined_confidence_decision(self, logits, alpha_val, exit_idx,
                                   max_weight=0.6, entropy_weight=0.4, adaptive_coeff=0.3):
        max_confidence = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        entropy_confidence = self.compute_entropy_confidence(logits, 'normalized_entropy')
        combined_confidence = max_weight * max_confidence + entropy_weight * entropy_confidence
        base_threshold = 0.5 + 0.2 * exit_idx / self.n_exits
        dynamic_threshold = max(0.2, min(0.9, base_threshold - adaptive_coeff * alpha_val))
        return combined_confidence > dynamic_threshold

    def clear_analysis_data(self):
        self.alpha_values = []
        self.exit_decisions_log = []
        self.policy_decisions_log = []
        self.cost_analysis_log = []

    def get_analysis_data(self):
        return {
            'alpha_values': self.alpha_values,
            'exit_decisions_log': self.exit_decisions_log,
            'policy_decisions_log': self.policy_decisions_log,
            'cost_analysis_log': self.cost_analysis_log
        }

    def save_adaptive_state(self, filepath):
        if hasattr(self, 'adaptive_coeff_manager'):
            self.adaptive_coeff_manager.save_online_learning_state(filepath)

    def load_adaptive_state(self, filepath):
        if hasattr(self, 'adaptive_coeff_manager'):
            return self.adaptive_coeff_manager.load_online_learning_state(filepath)
        return False
    
    def get_online_learning_stats(self):
        
        if hasattr(self, 'adaptive_coeff_manager'):
            return self.adaptive_coeff_manager.get_online_learning_stats()
        return {}
    
    def inference_with_online_learning(self, x, labels=None):
        
        
        was_training = self.training_mode
        self.training_mode = False
        
        
        final_outputs, exit_points, computation_costs = self._forward_inference(x)
        
        
        if labels is not None and hasattr(self, 'adaptive_coeff_manager'):
            predictions = torch.argmax(final_outputs, dim=1)
            correct_mask = (predictions == labels)
            
            
            alpha_values = None
            if self.use_difficulty_scaling and hasattr(self, 'alpha_values') and self.alpha_values:
                alpha_values = self.alpha_values[-len(labels):]
            
            
            for i in range(len(labels)):
                if alpha_values is not None and i < len(alpha_values):
                    alpha_score = alpha_values[i]
                    exit_idx = exit_points[i].item() - 1  
                    predicted_class = predictions[i].item()
                    was_correct = correct_mask[i].item()
                    
                    
                    confidence = torch.softmax(final_outputs[i:i+1], dim=1).max().item()
                    
                    
                    self.adaptive_coeff_manager.online_learning_update(
                        exit_idx=exit_idx,
                        alpha_score=alpha_score,
                        predicted_class=predicted_class,
                        was_correct=was_correct,
                        confidence=confidence
                    )
        
        
        self.training_mode = was_training
        return final_outputs, exit_points, computation_costs


def calibrate_exit_times_branchynet_resnet18(model, device, loader, n_batches=10):
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Returning zeros.")
        return [0.0] * 4
    model.training_mode = False
    model.eval().to(device)
    n_batches = min(n_batches, len(loader))
    if n_batches == 0: return [0.0] * 4

    exit_times_ms = [0.0] * 4
    total_samples = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches: break
            images = images.to(device)
            total_samples += images.size(0)
            start_event = torch.cuda.Event(enable_timing=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            
            start_event.record()
            x = model.relu(model.bn1(model.conv1(images)))
            
            x = model.layer1(x); _ = model.branch1(x); events[0].record()
            x = model.layer2(x); _ = model.branch2(x); events[1].record()
            x = model.layer3(x); _ = model.branch3(x); events[2].record()
            x = model.layer4(x); x = model.avgpool(x); x = torch.flatten(x, 1); _ = model.final_classifier(x); events[3].record()
            
            torch.cuda.synchronize()
            for j in range(4):
                exit_times_ms[j] += start_event.elapsed_time(events[j])

    avg_exit_times_s = [(t / total_samples) / 1000.0 for t in exit_times_ms]
    print(f"Calibrated BranchyResNet18 exit times (s/sample): {avg_exit_times_s}")
    return avg_exit_times_s


def calibrate_exit_times_resnet18(model, device, loader, n_batches=10):
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Returning zeros.")
        return [0.0] * 5
    model.training_mode = False
    model.eval().to(device)
    n_batches = min(n_batches, len(loader))
    if n_batches == 0: return [0.0] * 5

    exit_times_ms = [0.0] * 5
    total_samples = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches: break
            images = images.to(device)
            total_samples += images.size(0)
            start_event = torch.cuda.Event(enable_timing=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            
            start_event.record()
            x = model.relu(model.bn1(model.conv1(images)))
            
            x = model.layer1(x); _ = model.exit1(x); events[0].record()
            x = model.layer2(x); _ = model.exit2(x); events[1].record()
            x = model.layer3(x); _ = model.exit3(x); events[2].record()
            x = model.layer4(x); _ = model.exit4(x); events[3].record()
            
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            _ = model.final_classifier(x)
            events[4].record()
            
            torch.cuda.synchronize()
            for j in range(5):
                exit_times_ms[j] += start_event.elapsed_time(events[j])

    avg_exit_times_s = [(t / total_samples) / 1000.0 for t in exit_times_ms]
    print(f"Calibrated ResNet-18 exit times (s/sample): {avg_exit_times_s}")
    return avg_exit_times_s


    


def load_datasets(dataset_name='cifar10', batch_size=64):
    ds = dataset_name.lower()
    if ds == 'cifar100':
        # CIFAR-100 statistics
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    else:  # CIFAR10 (default)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_branchynet_resnet18(model, train_loader, test_loader=None, num_epochs=20, learning_rate=0.001):
    
    branch_weights = [0.2, 0.3, 0.3, 0.2]  
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    device = next(model.parameters()).device
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        model.training_mode = True
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)  
                    
                    
                    total_loss = 0
                    for i, (output, weight) in enumerate(zip(outputs, branch_weights)):
                        branch_loss = criterion(output, labels)
                        total_loss += weight * branch_loss
                        
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                
                
                total_loss = 0
                for i, (output, weight) in enumerate(zip(outputs, branch_weights)):
                    branch_loss = criterion(output, labels)
                    total_loss += weight * branch_loss
                    
                total_loss.backward()
                optimizer.step()
            
            running_loss += total_loss.item()
        
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        
        if test_loader is not None:
            results = evaluate_branchynet_resnet18(model, test_loader)
            accuracy = results['accuracy']
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            print(f'Exit Distribution: {results["exit_percentages"]}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_branchynet_resnet18(model, test_loader):
    
    model.eval()
    model.training_mode = False
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    inference_times = []
    exit_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            outputs, exit_points = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
            unique_exits, counts = torch.unique(exit_points, return_counts=True)
            for exit_val, count in zip(unique_exits, counts):
                exit_counts[exit_val.item()] += count.item()
    
    accuracy = 100 * correct / total
    exit_percentages = {k: (v / total) * 100 for k, v in exit_counts.items()}

    
    calibrated_times = calibrate_exit_times_branchynet_resnet18(model, device, test_loader, n_batches=20)
    weighted_avg_time_s = sum(exit_percentages.get(i, 0) / 100.0 * t for i, t in enumerate(calibrated_times, 1))
    avg_inference_time = weighted_avg_time_s * 1000  # ms per sample
    
    return {
        'accuracy': accuracy,
        'inference_time': avg_inference_time,
        'exit_counts': exit_counts,
        'exit_percentages': exit_percentages
    }


def train_static_resnet18(model, train_loader, test_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    device = next(model.parameters()).device
    
    scaler = GradScaler() if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU path
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        accuracy, _ = evaluate_static_resnet18(model, test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return model

def train_full_layer_resnet18(model, train_loader, test_loader, num_epochs, learning_rate):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    device = next(model.parameters()).device
    
    scaler = GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        model.training_mode = True
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.train_step(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU path
                loss = model.train_step(images, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        results = evaluate_full_layer_resnet18(model, test_loader)
        accuracy = results['accuracy']
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            best_model_state = model.state_dict().copy()

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_static_resnet18(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total, inference_times = 0, 0, []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_time = (sum(inference_times) / len(test_loader.dataset)) * 1000
    return accuracy, avg_time


def evaluate_full_layer_resnet18(model, test_loader, save_analysis_data=False):
    
    model.eval()
    model.training_mode = False
    device = next(model.parameters()).device
    model.clear_analysis_data()
    
    correct, total = 0, 0
    exit_counts = {i: 0 for i in range(1, model.n_exits + 1)}
    total_comp_cost, total_energy_cost = 0.0, 0.0
    alpha_stats = {'all': [], 'correct': [], 'incorrect': []}
    exit_alpha_stats = {i: {'correct': [], 'incorrect': []} for i in range(1, model.n_exits + 1)}
    policy_effectiveness = {'correct_early_exits': 0, 'total_early_exits': 0}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            
            outputs, exit_points, comp_costs = model(images)
            
            total_comp_cost += comp_costs.sum().item()
            for p in exit_points:
                total_energy_cost += model.energy_costs[p.item() - 1]
                
            _, predicted = torch.max(outputs.data, 1)
            correct_mask = (predicted == labels)
            total += labels.size(0)
            correct += correct_mask.sum().item()
            
            unique_exits, counts = torch.unique(exit_points, return_counts=True)
            for val, count in zip(unique_exits, counts):
                exit_idx = val.item()
                exit_counts[exit_idx] += count.item()
                if exit_idx < model.n_exits:
                    policy_effectiveness['total_early_exits'] += count.item()
                    policy_effectiveness['correct_early_exits'] += (correct_mask & (exit_points == exit_idx)).sum().item()

            if model.use_difficulty_scaling and model.alpha_values:
                batch_alphas = model.alpha_values[-images.size(0):]
                alpha_stats['all'].extend(batch_alphas)
                for i in range(images.size(0)):
                    alpha, exit_idx = batch_alphas[i], exit_points[i].item()
                    if correct_mask[i]:
                        alpha_stats['correct'].append(alpha)
                        exit_alpha_stats[exit_idx]['correct'].append(alpha)
                    else:
                        alpha_stats['incorrect'].append(alpha)
                        exit_alpha_stats[exit_idx]['incorrect'].append(alpha)

    accuracy = 100 * correct / total if total > 0 else 0
    exit_percentages = {k: (v / total) * 100 for k, v in exit_counts.items()} if total > 0 else {}
    avg_comp_cost = total_comp_cost / total if total > 0 else 0
    avg_energy_cost = total_energy_cost / total if total > 0 else 0
    
    early_exit_accuracy = 100 * policy_effectiveness['correct_early_exits'] / max(1, policy_effectiveness['total_early_exits'])
    
    calibrated_times = calibrate_exit_times_resnet18(model, device, test_loader, n_batches=20)
    weighted_avg_time_s = sum(exit_percentages.get(i, 0) / 100.0 * t for i, t in enumerate(calibrated_times, 1))
    final_inference_time_ms = weighted_avg_time_s * 1000

    results = {
        'accuracy': accuracy, 'inference_time': final_inference_time_ms,
        'exit_percentages': exit_percentages, 'alpha_stats': alpha_stats,
        'exit_alpha_stats': exit_alpha_stats,
        'cost_analysis': {'avg_computation_cost': avg_comp_cost, 'avg_energy_cost': avg_energy_cost},
        'policy_effectiveness': {
            'early_exit_accuracy': early_exit_accuracy,
            'early_exit_percentage': 100 * policy_effectiveness['total_early_exits'] / max(1, total),
        }
    }
    if save_analysis_data:
        results['analysis_data'] = model.get_analysis_data()
    return results



class PowerMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            self.handle = None
        self.power_measurements = queue.Queue()
        self.is_monitoring = False

    def start_monitoring(self):
        if self.handle is None: return
        self.is_monitoring = True
        while not self.power_measurements.empty(): self.power_measurements.get()
        def monitor():
            while self.is_monitoring:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.power_measurements.put(power)
                    time.sleep(0.005)
                except pynvml.NVMLError: pass
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()

    def stop_monitoring(self):
        if self.handle is None: return []
        self.is_monitoring = False
        self.monitor_thread.join()
        measurements = []
        while not self.power_measurements.empty():
            measurements.append(self.power_measurements.get())
        return measurements


def measure_power_consumption(model, test_loader, num_samples=1000, device='cuda', sustained_duration=5.0, override_time_per_sample_ms=None):
    
    model.eval()
    model.to(device)
    power_monitor = PowerMonitor()

    
    print("Measuring baseline GPU power...")
    power_monitor.start_monitoring()
    time.sleep(2.0)  # Wait 2 seconds to get baseline
    baseline_data = power_monitor.stop_monitoring()
    baseline_power = np.mean(baseline_data) if baseline_data else 0
    print(f"Baseline GPU power: {baseline_power:.2f}W")

    
    inference_data = []
    total_samples_to_load = 0
    with torch.no_grad():
        for images, _ in test_loader:
            if total_samples_to_load >= num_samples:
                break
            inference_data.append(images.to(device))
            total_samples_to_load += images.size(0)

    if not inference_data:
        print("No data available for power measurement")
        return {'avg_power': 0, 'peak_power': 0, 'energy': 0, 'inference_time': 0}

    print(f"Running sustained inference on {total_samples_to_load} unique samples for at least {sustained_duration}s...")

    
    power_monitor.start_monitoring()
    start_time = time.time()
    inference_count = 0
    while (time.time() - start_time) < sustained_duration:
        for batch in inference_data:
            if hasattr(model, 'training_mode'):
                model.training_mode = False
            _ = model(batch)
            inference_count += batch.size(0)
            if (time.time() - start_time) >= sustained_duration:
                break
    
    end_time = time.time()
    power_data = power_monitor.stop_monitoring()

    if not power_data:
        print("Warning: No power data collected during sustained inference.")
        return {'avg_power': baseline_power, 'peak_power': baseline_power, 'energy': 0, 'inference_time': 0}

    
    actual_duration = end_time - start_time
    avg_power = np.mean(power_data)
    peak_power = np.max(power_data)
    incremental_power = max(0, avg_power - baseline_power)
    total_energy = incremental_power * actual_duration
    
    if override_time_per_sample_ms is not None and override_time_per_sample_ms > 0:
        time_per_sample = override_time_per_sample_ms / 1000.0
        energy_per_sample = incremental_power * time_per_sample
    else:
        energy_per_sample = total_energy / inference_count if inference_count > 0 else 0
        time_per_sample = actual_duration / inference_count if inference_count > 0 else 0

    print(f"  Sustained avg power: {avg_power:.2f}W, Incremental power: {incremental_power:.2f}W")
    print(f"  Energy per sample: {energy_per_sample*1000:.4f}mJ")

    result_dict = {
        'avg_power': avg_power,
        'peak_power': peak_power,
        'energy': energy_per_sample,
        'inference_time': time_per_sample * 1000,
    }
    
    
    print(f"  DEBUG: returning avg_power={result_dict['avg_power']:.2f}W, peak_power={result_dict['peak_power']:.2f}W")
    
    return result_dict


def create_output_directory(dataset_name):
    output_dir = f'plots_{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_alpha_analysis(results, dataset_name):
    
    if 'analysis_results' not in results or 'alpha_stats' not in results['analysis_results']:
        return
    output_dir = create_output_directory(dataset_name)
    alpha_stats = results['analysis_results']['alpha_stats']
    exit_alpha_stats = results['analysis_results']['exit_alpha_stats']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{dataset_name.upper()} - Difficulty Score (α) Analysis', fontsize=16)

    
    ax1 = axes[0, 0]
    if alpha_stats['correct'] and alpha_stats['incorrect']:
        ax1.hist(alpha_stats['correct'], bins=30, alpha=0.7, label='Correct', color='green', density=True)
        ax1.hist(alpha_stats['incorrect'], bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
        ax1.legend()
    ax1.set_title('α Distribution: Correct vs Incorrect')
    
    
    ax2 = axes[0, 1]
    for exit_idx, data in exit_alpha_stats.items():
        ax2.scatter(data['correct'], [exit_idx] * len(data['correct']), c='green', alpha=0.5, s=10)
        ax2.scatter(data['incorrect'], [exit_idx] * len(data['incorrect']), c='red', alpha=0.5, s=10)
    ax2.set_title('α vs Exit Point')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_alpha_analysis.png'))
    plt.close()


def plot_joint_policy_analysis(results, dataset_name):
    
    if ('analysis_results' not in results or 'analysis_data' not in results['analysis_results'] or
        'policy_decisions_log' not in results['analysis_results']['analysis_data']):
        print("Warning: No joint policy decision data available for plotting.")
        return

    output_dir = create_output_directory(dataset_name)
    policy_data = results['analysis_results']['analysis_data']['policy_decisions_log']

    if not policy_data:
        print("Warning: Policy decision log is empty.")
        return

    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{dataset_name.upper()} - Joint Exit Policy Decision Boundaries', fontsize=16)
    axes = axes.flatten()

    for exit_idx in range(5):
        ax = axes[exit_idx]
        
        exit_decisions = [d for d in policy_data if d.get('exit_idx') == exit_idx]

        if exit_decisions:
            alphas = [d['alpha'] for d in exit_decisions]
            confidences = [d['confidence'] for d in exit_decisions]
            
            colors = ['green' if d['decision'] else 'red' for d in exit_decisions]
            
            ax.scatter(alphas, confidences, c=colors, alpha=0.5, s=15, edgecolors='none')
            ax.set_title(f'Exit {exit_idx + 1} Decisions')
            ax.set_xlabel('Difficulty Score (α)')
            ax.set_ylabel('Confidence Score')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Exit {exit_idx + 1} Decisions')
    
    
    axes[5].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_joint_policy_analysis.png'))
    plt.close()


def plot_exit_decision_heatmap(results, dataset_name):
    
    if ('analysis_results' not in results or 'analysis_data' not in results['analysis_results'] or
        'policy_decisions_log' not in results['analysis_results']['analysis_data']):
        print("Warning: No policy decision data available for heatmap.")
        return

    output_dir = create_output_directory(dataset_name)
    policy_data = results['analysis_results']['analysis_data']['policy_decisions_log']

    if not policy_data:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle(f'{dataset_name.upper()} - Exit Decision Heatmaps (α vs. Confidence)', fontsize=16)
    axes = axes.flatten()

    alpha_bins = np.linspace(0, 1, 15)
    conf_bins = np.linspace(0, 1, 15)

    for exit_idx in range(5):
        ax = axes[exit_idx]
        exit_decisions = [d for d in policy_data if d.get('exit_idx') == exit_idx]
        
        if exit_decisions:
            alphas = [d['alpha'] for d in exit_decisions]
            confs = [d['confidence'] for d in exit_decisions]
            decisions = [1 if d['decision'] else 0 for d in exit_decisions]

            
            exit_counts, _, _ = np.histogram2d(alphas, confs, bins=[alpha_bins, conf_bins], weights=decisions)
            total_counts, _, _ = np.histogram2d(alphas, confs, bins=[alpha_bins, conf_bins])
            
            
            with np.errstate(divide='ignore', invalid='ignore'):
                prob_matrix = np.divide(exit_counts, total_counts, out=np.zeros_like(exit_counts), where=total_counts!=0)
            
            im = ax.imshow(prob_matrix.T, cmap='RdYlGn', aspect='auto', extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=1)
            ax.set_title(f'Exit {exit_idx + 1} Probability')
        else:
             ax.set_title(f'Exit {exit_idx + 1} (No Data)')

    axes[5].axis('off') # Hide unused subplot
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Probability of Exiting')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_exit_decision_heatmap.png'))
    plt.close()


def analyze_exit_distribution(model, test_loader, dataset_name='cifar10'):
    
    device = next(model.parameters()).device
    model.eval()
    model.training_mode = False
    
    exit_counts = {i: 0 for i in range(1, model.n_exits + 1)}
    class_distributions = {i: defaultdict(int) for i in range(1, model.n_exits + 1)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, exit_points, _ = model(images)
            
            for exit_idx in range(1, model.n_exits + 1):
                mask = (exit_points == exit_idx)
                exit_counts[exit_idx] += mask.sum().item()
                for label in labels[mask]:
                    class_distributions[exit_idx][label.item()] += 1
    
    return exit_counts, class_distributions


def plot_class_distribution(class_distributions, dataset_name):
    
    output_dir = create_output_directory(dataset_name)
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    exit_indices = sorted(class_distributions.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    fig.suptitle(f'{dataset_name.upper()} - Class Distribution Across Exits', fontsize=16)
    axes = axes.flatten()

    for idx, exit_idx in enumerate(exit_indices):
        ax = axes[idx]
        distribution = class_distributions[exit_idx]
        labels = sorted(distribution.keys())
        counts = [distribution[k] for k in labels]
        
        ax.bar([class_names[i] for i in labels], counts, color=plt.cm.viridis(idx / len(exit_indices)))
        ax.set_title(f'Exit {exit_idx} Class Distribution')
        ax.tick_params(axis='x', rotation=45)
    
    for i in range(len(exit_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_class_distribution.png'))
    plt.close()


def save_alpha_data(results, dataset_name):
    
    output_dir = create_output_directory(dataset_name)
    analysis_data = results.get('analysis_results', {})
    
    
    alpha_stats = analysis_data.get('alpha_stats', {})
    if alpha_stats:
        np.save(os.path.join(output_dir, f'{dataset_name.lower()}_alpha_all.npy'), np.array(alpha_stats.get('all', [])))
        
    
    if 'analysis_data' in analysis_data and 'policy_decisions_log' in analysis_data['analysis_data']:
        policy_log = analysis_data['analysis_data']['policy_decisions_log']
        with open(os.path.join(output_dir, f'{dataset_name.lower()}_policy_log.json'), 'w') as f:
            json.dump(policy_log, f, indent=2)


def plot_cost_analysis(results, dataset_name):
    
    if 'analysis_results' not in results or 'cost_analysis' not in results['analysis_results']:
        print("Warning: Cost analysis data not found for plotting.")
        return

    output_dir = create_output_directory(dataset_name)
    analysis_data = results['analysis_results']
    cost_data = analysis_data['cost_analysis']
    exit_percentages = analysis_data['exit_percentages']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{dataset_name.upper()} - Cost-Aware Optimization Analysis', fontsize=16)

    
    ax1 = axes[0]
    exits = sorted(exit_percentages.keys())
    percentages = [exit_percentages[k] for k in exits]
    
    comp_costs = [0.2, 0.45, 0.7, 0.9, 1.0][:len(exits)]

    ax1.bar(exits, percentages, color='skyblue', label='Exit Usage')
    ax1.set_xlabel('Exit Point')
    ax1.set_ylabel('Percentage of Samples (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(exits, comp_costs, 'ro-', linewidth=2, markersize=6, label='Comp. Cost')
    ax2.set_ylabel('Relative Computation Cost', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.1)
    ax1.set_title('Exit Distribution vs. Computation Cost')

    
    ax3 = axes[1]
    policy_stats = analysis_data.get('policy_effectiveness', {})
    metrics = ['Early Exit\nAccuracy', 'Early Exit\nPercentage', 'Total\nAccuracy']
    
    
    values = [
        policy_stats.get('early_exit_accuracy', 0),
        policy_stats.get('early_exit_percentage', 0),
        analysis_data.get('accuracy', 0) 
    ]
    
    bars = ax3.bar(metrics, values, color=['orange', 'purple', 'cyan'], alpha=0.8)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Policy Effectiveness Metrics')
    ax3.set_ylim(0, 100)
    ax3.bar_label(bars, fmt='%.1f%%', padding=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_cost_analysis.png'))
    plt.close()


def plot_exit_distribution(exit_percentages, dataset_name):
    
    output_dir = create_output_directory(dataset_name)
    exits = list(exit_percentages.keys())
    percentages = list(exit_percentages.values())
    cmap = plt.cm.get_cmap('viridis')
    colors = [cmap(i / (len(exits) -1)) for i in range(len(exits))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'Exit {i}' for i in exits], percentages, color=colors)
    plt.ylabel('Percentage of Samples (%)')
    plt.title(f'{dataset_name.upper()} - Exit Distribution')
    for bar, perc in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{perc:.1f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_exit_distribution.png'))
    plt.close()


def plot_comparative_analysis(static_results, early_exit_results, dataset_name):
    
    output_dir = create_output_directory(dataset_name)
    metrics = ['Accuracy (%)', 'Inference Time (ms)', 'Energy/Sample (mJ)']
    static_vals = [
        static_results['accuracy'], 
        static_results['inference_time'], 
        static_results['power']['energy'] * 1000
    ]
    ee_vals = [
        early_exit_results['accuracy'],
        early_exit_results['inference_time'],
        early_exit_results['power']['energy'] * 1000
    ]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, static_vals, width, label='Static ResNet-18', color='#2ecc71')
    rects2 = ax.bar(x + width/2, ee_vals, width, label='Dynamic ResNet-18', color='#3498db')

    ax.set_ylabel('Scores')
    ax.set_title(f'Static vs. Dynamic ResNet-18 on {dataset_name.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_comparative_analysis.png'))
    plt.close()


def run_comprehensive_comparison_experiments_resnet18(dataset_name):
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EARLY EXIT COMPARISON ON {dataset_name.upper()}")
    print(f"Comparing: Static ResNet18 | BranchyResNet18 | Our Full-Layer Framework")
    print(f"{'='*80}")
    
    train_loader, test_loader = load_datasets(dataset_name, batch_size=32)
    in_channels = 3
    num_classes = 100 if dataset_name.lower() == 'cifar100' else 10
    weights_dir = 'pretrained_weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    static_weights_path = os.path.join(weights_dir, f'static_resnet18_{dataset_name.lower()}.pth')
    branchynet_weights_path = os.path.join(weights_dir, f'branchynet_resnet18_{dataset_name.lower()}.pth')
    full_layer_weights_path = os.path.join(weights_dir, f'full_layer_resnet18_{dataset_name.lower()}.pth')
    policy_path = os.path.join(weights_dir, f'full_layer_resnet18_{dataset_name.lower()}_joint_policy.npz')
    
    
    
    print(f"\n[1/4] Setting up Static ResNet18 baseline...")
    static_resnet18 = StaticResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    
    if os.path.exists(static_weights_path):
        print("Loading pre-trained Static ResNet18 weights...")
        checkpoint = torch.load(static_weights_path, map_location=device)
        static_resnet18.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights with validation accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Training Static ResNet18 from scratch...")
        static_resnet18 = train_static_resnet18(static_resnet18, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
        acc, _ = evaluate_static_resnet18(static_resnet18, test_loader)
        torch.save({
            'state_dict': static_resnet18.state_dict(),
            'accuracy': acc,
            'training_time': time.time() - time.time(),  # Training time would be captured during actual training
            'model_params': sum(p.numel() for p in static_resnet18.parameters())
        }, static_weights_path)
    

    print("\nEvaluating Static ResNet18 baseline...")
    static_accuracy, static_inference_time = evaluate_static_resnet18(static_resnet18, test_loader)
    static_power = measure_power_consumption(static_resnet18, test_loader, num_samples=100, override_time_per_sample_ms=static_inference_time)
    
    print(f"Static ResNet18 Results:")
    print(f"  Accuracy: {static_accuracy:.2f}%")
    print(f"  Inference Time: {static_inference_time:.2f} ms")
    print(f"  Energy per Sample: {static_power['energy']*1000:.2f} mJ")
    

    print(f"\n[2/4] Setting up BranchyResNet18...")
    threshold = 0.35 if dataset_name.lower() == 'cifar100' else 0.30
    branchynet_resnet18 = BranchyResNet18(num_classes=num_classes, in_channels=in_channels, exit_threshold=threshold).to(device)
    print(f"Using entropy threshold: {threshold} for {dataset_name.upper()}")
    
    if os.path.exists(branchynet_weights_path):
        print("Loading pre-trained BranchyResNet18 weights...")
        checkpoint = torch.load(branchynet_weights_path, map_location=device)
        branchynet_resnet18.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights with validation accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Training BranchyResNet18 from scratch...")
        branchynet_resnet18 = train_branchynet_resnet18(branchynet_resnet18, train_loader, test_loader, num_epochs=20)
        final_results = evaluate_branchynet_resnet18(branchynet_resnet18, test_loader)
        torch.save({
            'state_dict': branchynet_resnet18.state_dict(),
            'accuracy': final_results['accuracy'],
            'training_time': time.time() - time.time(),  
            'model_params': sum(p.numel() for p in branchynet_resnet18.parameters()),
            'exit_percentages': final_results.get('exit_percentages', {}),
            'inference_time': final_results.get('inference_time', 0.0)
        }, branchynet_weights_path)
    
    print("\nEvaluating BranchyResNet18...")
    branchynet_results = evaluate_branchynet_resnet18(branchynet_resnet18, test_loader)
    branchynet_power = measure_power_consumption(branchynet_resnet18, test_loader, num_samples=100, override_time_per_sample_ms=branchynet_results['inference_time'])
    
    print(f"BranchyResNet18 Results:")
    print(f"  Accuracy: {branchynet_results['accuracy']:.2f}%")
    print(f"  Inference Time: {branchynet_results['inference_time']:.2f} ms")
    print(f"  Energy per Sample: {branchynet_power['energy']*1000:.2f} mJ")
    print(f"  Exit Distribution: {branchynet_results['exit_percentages']}")
    
    print(f"\n[3/4] Setting up Our Full-Layer Framework...")
    full_layer_resnet18 = FullLayerResNet18(
        num_classes=num_classes, 
        in_channels=in_channels,
        use_difficulty_scaling=True,
        use_joint_policy=True,
        use_cost_awareness=True
    ).to(device)
    
    if os.path.exists(full_layer_weights_path):
        print("Loading pre-trained FullLayerResNet18 weights...")
        checkpoint = torch.load(full_layer_weights_path, map_location=device)
        full_layer_resnet18.load_state_dict(checkpoint['state_dict'])
        if os.path.exists(policy_path):
            policy_data = np.load(policy_path)
            full_layer_resnet18.joint_policy.value_table = policy_data['value_table']
            full_layer_resnet18.joint_policy.policy_table = policy_data['policy_table']
            full_layer_resnet18.joint_policy.is_trained = True
        print(f"Loaded weights with validation accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Training FullLayerResNet18 from scratch...")
        full_layer_resnet18 = train_full_layer_resnet18(full_layer_resnet18, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
        res = evaluate_full_layer_resnet18(full_layer_resnet18, test_loader)
        torch.save({
            'state_dict': full_layer_resnet18.state_dict(),
            'accuracy': res['accuracy']
        }, full_layer_weights_path)
        if full_layer_resnet18.joint_policy and full_layer_resnet18.joint_policy.is_trained:
            np.savez(policy_path, 
                value_table=full_layer_resnet18.joint_policy.value_table, 
                policy_table=full_layer_resnet18.joint_policy.policy_table, 
                is_trained=True)
    
    print("\nEvaluating Our Full-Layer Framework...")
    full_layer_results = evaluate_full_layer_resnet18(full_layer_resnet18, test_loader, save_analysis_data=True)
    full_layer_power = measure_power_consumption(full_layer_resnet18, test_loader, num_samples=100, override_time_per_sample_ms=full_layer_results['inference_time'])
    
    print(f"Our Full-Layer Framework Results:")
    print(f"  Accuracy: {full_layer_results['accuracy']:.2f}%")
    print(f"  Inference Time: {full_layer_results['inference_time']:.2f} ms")
    print(f"  Energy per Sample: {full_layer_power['energy']*1000:.2f} mJ")
    print(f"  Exit Distribution: {full_layer_results['exit_percentages']}")
    print(f"  Early Exit Accuracy: {full_layer_results['policy_effectiveness']['early_exit_accuracy']:.2f}%")
    
    print(f"\n[4/4] Generating comprehensive analysis and visualizations...")
    
    all_results = {
        'static': {
            'accuracy': static_accuracy,
            'inference_time': static_inference_time,
            'power': static_power
        },
        'branchynet': {
            **branchynet_results,
            'power': branchynet_power
        },
        'full_layer': {
            **full_layer_results,
            'power': full_layer_power
        }
    }
    
    plot_comprehensive_comparison(all_results, dataset_name)
    plot_exit_distribution(full_layer_results['exit_percentages'], dataset_name)
    plot_alpha_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_cost_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_joint_policy_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_exit_decision_heatmap({'analysis_results': full_layer_results}, dataset_name)
    
    _, class_dist = analyze_exit_distribution(full_layer_resnet18, test_loader, dataset_name)
    plot_class_distribution(class_dist, dataset_name)
    save_alpha_data({'analysis_results': full_layer_results}, dataset_name)
    
    results_data = {
        'static': all_results['static'],
        'branchynet': all_results['branchynet'],
        'full_layer': all_results['full_layer'],
        'comparison': {
            'static_accuracy': all_results['static']['accuracy'],
            'branchynet_accuracy': all_results['branchynet']['accuracy'], 
            'full_layer_accuracy': all_results['full_layer']['accuracy'],
            'static_inference_time': all_results['static']['inference_time'],
            'branchynet_inference_time': all_results['branchynet']['inference_time'],
            'full_layer_inference_time': all_results['full_layer']['inference_time'],
            'energy_savings_branchy': ((all_results['static']['power']['energy'] - all_results['branchynet']['power']['energy']) / all_results['static']['power']['energy']) * 100,
            'energy_savings_full_layer': ((all_results['static']['power']['energy'] - all_results['full_layer']['power']['energy']) / all_results['static']['power']['energy']) * 100,
        }
    }
    
    results_path = f'results/resnet18_comprehensive_results_{dataset_name.lower()}.pkl'
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Comprehensive results saved to {results_path}")
    
    json_results_path = f'results/resnet18_comprehensive_results_{dataset_name.lower()}.json'
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_compatible_results = json.loads(json.dumps(results_data, default=convert_numpy))
    with open(json_results_path, 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    print(f"Comprehensive results saved to JSON: {json_results_path}")
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE COMPARISON COMPLETED FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    
    return all_results


def plot_comprehensive_comparison(all_results, dataset_name):
    
    output_dir = create_output_directory(dataset_name)
    
    
    methods = ['Static ResNet18', 'BranchyNet ResNet18', 'Full-Layer ResNet18']
    accuracies = [
        all_results['static']['accuracy'],
        all_results['branchynet']['accuracy'], 
        all_results['full_layer']['accuracy']
    ]
    inference_times = [
        all_results['static']['inference_time'],
        all_results['branchynet']['inference_time'],
        all_results['full_layer']['inference_time']
    ]
    
    print(f"DEBUG - Static power dict: {all_results['static']['power']}")
    print(f"DEBUG - BranchyNet power dict: {all_results['branchynet']['power']}")
    print(f"DEBUG - Full-layer power dict: {all_results['full_layer']['power']}")
    
    avg_powers = [
        all_results['static']['power'].get('avg_power', 0),
        all_results['branchynet']['power'].get('avg_power', 0),
        all_results['full_layer']['power'].get('avg_power', 0)
    ]
    peak_powers = [
        all_results['static']['power'].get('peak_power', 0),
        all_results['branchynet']['power'].get('peak_power', 0),
        all_results['full_layer']['power'].get('peak_power', 0)
    ]
    
    print(f"DEBUG - avg_powers extracted: {avg_powers}")
    print(f"DEBUG - peak_powers extracted: {peak_powers}")
    energies = [
        all_results['static']['power']['energy'] * 1000,
        all_results['branchynet']['power']['energy'] * 1000,
        all_results['full_layer']['power']['energy'] * 1000
    ]
    
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    colors = ['#3498db', '#e74c3c', '#27ae60']  # Blue, Red, Green
    
    fig.suptitle(f'{dataset_name.upper()} - Comprehensive Model Comparison', fontsize=16)
    
    
    ax1 = axes[0, 0]
    bars = ax1.bar(methods, accuracies, color=colors, width=0.6)
    ax1.set_title('Accuracy Comparison', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.tick_params(axis='x', rotation=15)

    
    ax2 = axes[0, 1]
    bars = ax2.bar(methods, inference_times, color=colors, width=0.6)
    ax2.set_title('Inference Time', fontsize=14)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.tick_params(axis='x', rotation=15)
    
    
    ax3 = axes[0, 2]
    bars = ax3.bar(methods, avg_powers, color=colors, width=0.6)
    ax3.set_title('Average Power Consumption', fontsize=14)
    ax3.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=10)
    ax3.tick_params(axis='x', rotation=15)
    
    
    ax4 = axes[1, 0]
    bars = ax4.bar(methods, peak_powers, color=colors, width=0.6)
    ax4.set_title('Peak Power Consumption', fontsize=14)
    ax4.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=10)
    ax4.tick_params(axis='x', rotation=15)
    
    
    ax5 = axes[1, 1]
    bars = ax5.bar(methods, energies, color=colors, width=0.6)
    ax5.set_title('Energy Consumption', fontsize=14)
    ax5.set_ylabel('Energy (mJ)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=10)
    ax5.tick_params(axis='x', rotation=15)
    
    
    ax6 = axes[1, 2]
    for i, (acc, energy, method) in enumerate(zip(accuracies, energies, methods)):
        ax6.scatter(acc, energy, color=colors[i], s=200, alpha=0.7, label=method)
        ax6.text(acc + 0.2, energy, method.replace(' ResNet18', ''), fontsize=9, ha='left')
    ax6.set_xlabel('Accuracy (%)', fontsize=12)
    ax6.set_ylabel('Energy (mJ)', fontsize=12)
    ax6.set_title('Energy vs Accuracy Trade-off', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_three_model_comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive comparison plot saved: {dataset_name.lower()}_three_model_comprehensive_comparison.png")


def run_full_layer_experiments(dataset_name):
    
    print(f"\n{'='*60}\nRUNNING FULL-LAYER RESNET-18 EXPERIMENTS ON {dataset_name.upper()}\n{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_datasets(dataset_name, batch_size=128)
    weights_dir = 'pretrained_weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    static_weights_path = os.path.join(weights_dir, f'static_resnet18_{dataset_name.lower()}.pth')
    full_layer_weights_path = os.path.join(weights_dir, f'full_layer_resnet18_{dataset_name.lower()}.pth')
    policy_path = os.path.join(weights_dir, f'full_layer_resnet18_{dataset_name.lower()}_joint_policy.npz')

    
    print("\n[1/4] Setting up Static ResNet-18 baseline...")
    static_resnet = StaticResNet18(num_classes=10).to(device)
    if not os.path.exists(static_weights_path):
        print("Training Static ResNet-18...")
        train_static_resnet18(static_resnet, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
        acc, _ = evaluate_static_resnet18(static_resnet, test_loader)
        torch.save({'state_dict': static_resnet.state_dict(), 'accuracy': acc}, static_weights_path)
    checkpoint = torch.load(static_weights_path, map_location=device)
    static_resnet.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded Static ResNet-18 (Acc: {checkpoint.get('accuracy', 'N/A'):.2f}%)")
    
    print("\nEvaluating Static ResNet-18 baseline...")
    static_accuracy, static_inference_time = evaluate_static_resnet18(static_resnet, test_loader)
    static_power = measure_power_consumption(static_resnet, test_loader)

    # Full-Layer ResNet-18
    print("\n[2/4] Setting up FullLayerResNet18...")
    full_layer_resnet = FullLayerResNet18(num_classes=10).to(device)
    if not os.path.exists(full_layer_weights_path):
        print("Training FullLayerResNet18...")
        train_full_layer_resnet18(full_layer_resnet, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
        res = evaluate_full_layer_resnet18(full_layer_resnet, test_loader)
        torch.save({
            'state_dict': full_layer_resnet.state_dict(),
            'accuracy': res['accuracy'],
            'training_time': time.time() - time.time(),  # Training time would be captured during actual training
            'model_params': sum(p.numel() for p in full_layer_resnet.parameters()),
            'exit_percentages': res.get('exit_percentages', {}),
            'inference_time': res.get('inference_time', 0.0),
            'policy_effectiveness': res.get('policy_effectiveness', {}),
            'training_config': {
                'use_difficulty_scaling': getattr(full_layer_resnet, 'use_difficulty_scaling', True),
                'use_joint_policy': getattr(full_layer_resnet, 'use_joint_policy', True),
                'use_cost_awareness': getattr(full_layer_resnet, 'use_cost_awareness', True),
                'n_exits': getattr(full_layer_resnet, 'n_exits', 5)
            }
        }, full_layer_weights_path)
        if full_layer_resnet.joint_policy and full_layer_resnet.joint_policy.is_trained:
            np.savez(policy_path, value_table=full_layer_resnet.joint_policy.value_table, policy_table=full_layer_resnet.joint_policy.policy_table, is_trained=True)
            print(f"Joint policy saved to {policy_path}")
            
    checkpoint = torch.load(full_layer_weights_path, map_location=device)
    full_layer_resnet.load_state_dict(checkpoint['state_dict'])
    if os.path.exists(policy_path):
        policy_data = np.load(policy_path)
        full_layer_resnet.joint_policy.value_table = policy_data['value_table']
        full_layer_resnet.joint_policy.policy_table = policy_data['policy_table']
        full_layer_resnet.joint_policy.is_trained = True
    print(f"Loaded FullLayerResNet18 (Acc: {checkpoint.get('accuracy', 'N/A'):.2f}%)")

    print("\n[3/4] Comprehensive evaluation...")
    full_layer_results = evaluate_full_layer_resnet18(full_layer_resnet, test_loader, save_analysis_data=True)
    full_layer_power = measure_power_consumption(full_layer_resnet, test_loader)
    
    speed_improvement = (static_inference_time - full_layer_results['inference_time']) / static_inference_time * 100 if static_inference_time > 0 else 0.0
    energy_savings = (static_power['energy'] - full_layer_power['energy']) / static_power['energy'] * 100 if static_power['energy'] > 0 else 0.0

    print("\n[4/4] Generating visualizations...")
    static_results_plot = {'accuracy': static_accuracy, 'inference_time': static_inference_time, 'power': static_power}
    full_layer_results_plot = {**full_layer_results, 'power': full_layer_power}

    # Call all plotting functions
    plot_comparative_analysis(static_results_plot, full_layer_results_plot, dataset_name)
    plot_exit_distribution(full_layer_results['exit_percentages'], dataset_name)
    plot_alpha_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_cost_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_joint_policy_analysis({'analysis_results': full_layer_results}, dataset_name)
    plot_exit_decision_heatmap({'analysis_results': full_layer_results}, dataset_name)
    
    _, class_dist = analyze_exit_distribution(full_layer_resnet, test_loader, dataset_name)
    plot_class_distribution(class_dist, dataset_name)
    save_alpha_data({'analysis_results': full_layer_results}, dataset_name)
    
    print(f"\n{'='*60}\nEXPERIMENTS COMPLETED FOR {dataset_name.upper()}\n{'='*60}")

    return {
        'static': static_results_plot,
        'full_layer': full_layer_results_plot,
        'improvements': {'speed': speed_improvement, 'energy': energy_savings}
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EARLY EXIT RESNET-18 COMPARISON")
    print("="*80)
    
    dataset = os.environ.get('RESNET_DATASET', 'cifar10')
    cifar_results = run_comprehensive_comparison_experiments_resnet18(dataset)

    print("\n\n" + "="*80)
    print(f"FINAL COMPREHENSIVE RESULTS SUMMARY FOR {dataset.upper()}")
    print("="*80)
    
    static_res = cifar_results['static']
    branchy_res = cifar_results['branchynet']
    full_layer_res = cifar_results['full_layer']
    
    print("MODEL ACCURACY COMPARISON:")
    print(f"Static ResNet-18 (Baseline):")
    print(f"  Accuracy: {static_res['accuracy']:.2f}%")
    print(f"  Inference: {static_res['inference_time']:.2f}ms")
    print(f"  Energy: {static_res['power']['energy']*1000:.2f}mJ")
    
    print(f"\nBranchyNet ResNet-18 (4 exits):")
    print(f"  Accuracy: {branchy_res['accuracy']:.2f}%")
    print(f"  Inference: {branchy_res['inference_time']:.2f}ms")
    print(f"  Energy: {branchy_res['power']['energy']*1000:.2f}mJ")
    print(f"  Exit Distribution: {branchy_res['exit_percentages']}")
    
    print(f"\nOur Full-Layer ResNet-18 (5 exits):")
    print(f"  Accuracy: {full_layer_res['accuracy']:.2f}%")
    print(f"  Inference: {full_layer_res['inference_time']:.2f}ms")
    print(f"  Energy: {full_layer_res['power']['energy']*1000:.2f}mJ")
    print(f"  Exit Distribution: {full_layer_res['exit_percentages']}")
    print(f"  Early Exit Accuracy: {full_layer_res['policy_effectiveness']['early_exit_accuracy']:.2f}%")
    
    print("\nPOWER CONSUMPTION COMPARISON:")
    print(f"Static ResNet-18 - Avg: {static_res['power'].get('avg_power', 0):.2f}W, Peak: {static_res['power'].get('peak_power', 0):.2f}W, Energy: {static_res['power']['energy']:.4f}J")
    print(f"BranchyNet ResNet-18 - Avg: {branchy_res['power'].get('avg_power', 0):.2f}W, Peak: {branchy_res['power'].get('peak_power', 0):.2f}W, Energy: {branchy_res['power']['energy']:.4f}J")
    print(f"Full-Layer ResNet-18 - Avg: {full_layer_res['power'].get('avg_power', 0):.2f}W, Peak: {full_layer_res['power'].get('peak_power', 0):.2f}W, Energy: {full_layer_res['power']['energy']:.4f}J")
    
    speed_improvement_branchy = (static_res['inference_time'] - branchy_res['inference_time']) / static_res['inference_time'] * 100 if static_res['inference_time'] > 0 else 0.0
    speed_improvement_full = (static_res['inference_time'] - full_layer_res['inference_time']) / static_res['inference_time'] * 100 if static_res['inference_time'] > 0 else 0.0
    
    energy_savings_branchy = (static_res['power']['energy'] - branchy_res['power']['energy']) / static_res['power']['energy'] * 100 if static_res['power']['energy'] > 0 else 0.0
    energy_savings_full = (static_res['power']['energy'] - full_layer_res['power']['energy']) / static_res['power']['energy'] * 100 if static_res['power']['energy'] > 0 else 0.0
    
    print(f"\nImprovement Analysis:")
    print(f"  BranchyNet vs Static:")
    print(f"    Speed Improvement: {speed_improvement_branchy:+.1f}%")
    print(f"    Energy Savings: {energy_savings_branchy:+.1f}%")
    print(f"  Our Framework vs Static:")
    print(f"    Speed Improvement: {speed_improvement_full:+.1f}%")
    print(f"    Energy Savings: {energy_savings_full:+.1f}%")

    print("\n" + "="*80)
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("📊 Check the plots_cifar10/ directory for visualizations.")
    print("="*80)