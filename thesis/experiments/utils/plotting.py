import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import os
from pathlib import Path

# Set up the modern style
plt.style.use('seaborn-v0_8-white')
sns.set_style("white")
sns.set_context("notebook", font_scale=1.2)

# Beautiful color palette (same as before, but adapted for matplotlib)
COLORS = {
    'train': '#0EA5E9',  # vivid sky blue
    'val': '#F97316',    # warm orange
    'metric': '#10B981', # emerald green
    'lr': '#8B5CF6',     # rich purple
    'background': '#FAFAFA',
    'grid': '#E5E7EB',
    'text': '#1F2937'
}

def prepare_data(metrics: List[Dict[str, Union[float, int]]], window_size: int = 5) -> pd.DataFrame:
    """Prepare and smooth metrics data for plotting."""
    df = pd.DataFrame(metrics)
    metric_cols = [
        col for col in df.columns 
        if col not in ["step", "epoch"] and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    for col in metric_cols:
        df[f"{col}_smooth"] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f"{col}_std"] = df[col].rolling(window=window_size, min_periods=1).std()
        df[f"{col}_upper"] = df[f"{col}_smooth"] + df[f"{col}_std"]
        df[f"{col}_lower"] = df[f"{col}_smooth"] - df[f"{col}_std"]
    
    return df

def setup_figure_style(fig, ax, title: str, ylabel: str):
    """Apply consistent, beautiful styling to the figure."""
    # Set background colors
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # Title and label styling
    ax.set_title(title, pad=20, fontsize=16, color=COLORS['text'], 
                fontfamily='sans-serif', fontweight='bold', loc='left')
    ax.set_xlabel('Steps', fontsize=12, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=12, color=COLORS['text'])
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)
    
    # Tick styling
    ax.tick_params(colors=COLORS['text'])
    
    return fig, ax

def create_loss_plot(df: pd.DataFrame, is_llm: bool = False, figsize=(10, 6)):
    """Create a beautiful loss plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if "train_loss" in df.columns:
        # Training loss
        ax.plot(df["step"], df["train_loss_smooth"], 
                color=COLORS['train'], label='Training Loss', linewidth=2)
        ax.fill_between(df["step"], df["train_loss_lower"], df["train_loss_upper"],
                       color=COLORS['train'], alpha=0.1)
    
    if "val_loss" in df.columns:
        # Validation loss
        ax.plot(df["step"], df["val_loss_smooth"],
                color=COLORS['val'], label='Validation Loss', linewidth=2)
        ax.fill_between(df["step"], df["val_loss_lower"], df["val_loss_upper"],
                       color=COLORS['val'], alpha=0.1)
    
    if is_llm:
        ax.set_yscale('log')
    
    # Style the plot
    fig, ax = setup_figure_style(fig, ax, "Training and Validation Loss", "Loss")
    
    # Legend styling
    ax.legend(frameon=True, fancybox=True, framealpha=0.9,
             edgecolor=COLORS['grid'], loc='upper right')
    
    plt.tight_layout()
    return fig

def create_combined_plot(df: pd.DataFrame, task_type: str = "synthetic", figsize=(12, 15)):
    """Create a beautiful combined metrics plot."""
    is_llm = task_type.lower() == "llm"
    metric_name = "perplexity" if is_llm else "accuracy"
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 2, 1])
    fig.patch.set_facecolor(COLORS['background'])
    
    # Loss plot
    if "train_loss" in df.columns:
        axes[0].plot(df["step"], df["train_loss_smooth"],
                    color=COLORS['train'], label='Training Loss', linewidth=2)
        axes[0].fill_between(df["step"], df["train_loss_lower"], df["train_loss_upper"],
                           color=COLORS['train'], alpha=0.1)
    
    if "val_loss" in df.columns:
        axes[0].plot(df["step"], df["val_loss_smooth"],
                    color=COLORS['val'], label='Validation Loss', linewidth=2)
        axes[0].fill_between(df["step"], df["val_loss_lower"], df["val_loss_upper"],
                           color=COLORS['val'], alpha=0.1)
    
    if is_llm:
        axes[0].set_yscale('log')
    
    # Metric plot
    metric_col = f"val_{metric_name}"
    if metric_col in df.columns or "val_acc" in df.columns:
        metric_col = metric_col if metric_col in df.columns else "val_acc"
        axes[1].plot(df["step"], df[f"{metric_col}_smooth"],
                    color=COLORS['metric'], label=f'Validation {metric_name.title()}', linewidth=2)
        axes[1].fill_between(df["step"], df[f"{metric_col}_lower"], df[f"{metric_col}_upper"],
                           color=COLORS['metric'], alpha=0.1)
    
    # Learning rate plot
    if "learning_rate" in df.columns:
        axes[2].plot(df["step"], df["learning_rate"],
                    color=COLORS['lr'], label='Learning Rate', linewidth=2)
        axes[2].set_yscale('log')
    
    # Style each subplot
    titles = ["Loss", metric_name.title(), "Learning Rate"]
    ylabels = ["Loss", metric_name.title(), "Learning Rate"]
    
    for ax, title, ylabel in zip(axes, titles, ylabels):
        setup_figure_style(fig, ax, title, ylabel)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9,
                 edgecolor=COLORS['grid'], loc='upper right')
    
    plt.tight_layout()
    return fig

def save_plots(df: pd.DataFrame, output_dir: str, task_type: str = "synthetic") -> None:
    """Save all plots with high DPI for crisp rendering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual plots
    create_loss_plot(df, is_llm=(task_type == "llm")).savefig(
        output_dir / "loss_plot.png", dpi=300, bbox_inches='tight'
    )
    
    # Save combined plot
    create_combined_plot(df, task_type=task_type).savefig(
        output_dir / "combined_plot.png", dpi=300, bbox_inches='tight'
    )
    
    plt.close('all')  # Clean up

def plot_experiment(
    metrics: List[Dict[str, Union[float, int]]], 
    output_dir: str, 
    task_type: str = "synthetic", 
    window_size: int = 5
) -> None:
    """Main function to create and save all plots for an experiment."""
    df = prepare_data(metrics, window_size=window_size)
    save_plots(df, output_dir, task_type=task_type)

def exists(output_dir):
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
