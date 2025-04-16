import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from palette import selected_palette  # Import the chosen palette

# --- Simulation Parameters ---
num_steps = 20000  # Simulate training steps
eval_points = 100  # How many points to plot for the curve
steps = np.linspace(0, num_steps, eval_points)

# Define models (order matching palette_vibrant_thesis, excluding DeltaNet)
model_names = [
    "Mamba",  # Green
    "GLA",  # Blue
    "RetNet",  # Cyan
    "RWKV7",  # Purple (Using 7 instead of 4)
    # "Hyena", # Orange - Let's replace with models we tested
    "Transformer",  # Use Orange slot
    "FlashSTU",  # Use Red Slot (originally DeltaNet)
    "Spectron",  # Use Magenta/Pink slot (need 7th color)
    "LinearAttention",  # Use Teal/another slot (need 8th color)
]

# Extend palette if needed (adding a pink and a brown/dark orange for example)
if len(model_names) > len(selected_palette):
    print("Extending palette for more models...")
    # Add more vibrant colors if necessary
    selected_palette.extend(
        [
            "#FF69B4",  # Hot Pink (for Spectron)
            "#A0522D",  # Sienna/Brown (for LinearAttention) - Or choose another distinct color
            # E.g., "#8A2BE2" # Blue Violet
        ]
    )
    # Ensure palette length matches model count
    selected_palette = selected_palette[: len(model_names)]


# --- Artificial Data Generation ---
# We'll use sigmoid-like functions scaled to 0-100% accuracy
# f(x) = max_acc / (1 + exp(-k * (x - x0))) + noise
def generate_learning_curve(steps, max_acc, steepness, midpoint_shift, noise_level=1.5):
    """Generates a plausible learning curve."""
    # Apply a slight curve to the steps input for more realistic non-linearity
    # Log shift ensures curve starts changing noticeably later for higher shifts
    log_shifted_steps = np.log(steps + 1) - midpoint_shift
    curve = max_acc / (1 + np.exp(-steepness * log_shifted_steps))
    # Add noise, ensuring it doesn't go wildly out of bounds
    noise = np.random.normal(0, noise_level, size=steps.shape)
    noisy_curve = np.clip(curve + noise, 0, 100)  # Clip between 0 and 100
    # Ensure it starts near 0
    noisy_curve[0] = max(0, noisy_curve[0] - curve[0] + np.random.uniform(0, 1))  # Start close to 0
    return noisy_curve


# Define different learning dynamics for each model
artificial_accuracies = {}
# Mamba: Learns fast, high accuracy
artificial_accuracies["Mamba"] = generate_learning_curve(
    steps, max_acc=99, steepness=0.8, midpoint_shift=np.log(num_steps * 0.2)
)
# GLA: Learns reasonably well, maybe slightly lower plateau
artificial_accuracies["GLA"] = generate_learning_curve(
    steps, max_acc=96, steepness=0.7, midpoint_shift=np.log(num_steps * 0.3)
)
# RetNet: Similar to GLA maybe slightly slower start
artificial_accuracies["RetNet"] = generate_learning_curve(
    steps, max_acc=95, steepness=0.65, midpoint_shift=np.log(num_steps * 0.35)
)
# RWKV7: Slower learner, lower plateau
artificial_accuracies["RWKV7"] = generate_learning_curve(
    steps, max_acc=75, steepness=0.5, midpoint_shift=np.log(num_steps * 0.5)
)
# Transformer: Steady learner, high accuracy
artificial_accuracies["Transformer"] = generate_learning_curve(
    steps, max_acc=98, steepness=0.7, midpoint_shift=np.log(num_steps * 0.4)
)
# FlashSTU: Maybe decent performance, moderate speed
artificial_accuracies["FlashSTU"] = generate_learning_curve(
    steps, max_acc=92, steepness=0.6, midpoint_shift=np.log(num_steps * 0.4)
)
# Spectron: Maybe learns fast initially but plateaus lower? Or make it strong. Let's make it strong.
artificial_accuracies["Spectron"] = generate_learning_curve(
    steps, max_acc=97, steepness=0.9, midpoint_shift=np.log(num_steps * 0.25)
)
# LinearAttention: Maybe struggles a bit more? Lower plateau.
artificial_accuracies["LinearAttention"] = generate_learning_curve(
    steps, max_acc=85, steepness=0.55, midpoint_shift=np.log(num_steps * 0.45)
)


# --- Plotting ---
sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--"})
plt.figure(figsize=(10, 7))  # Adjusted figure size slightly

# Map models to colors explicitly for consistency
color_map = {model: color for model, color in zip(model_names, selected_palette)}

# Plot each model's data
for model in model_names:
    plt.plot(
        steps,
        artificial_accuracies[model],
        # marker='', # No markers for smooth curve look
        linestyle="-",
        linewidth=3.5,  # Slightly thicker lines
        label=model,
        color=color_map[model],  # Use explicit color mapping
    )

# Customize plot labels, limits, ticks, etc.
plt.xlabel("Training Steps", fontsize=22, labelpad=15)
plt.ylabel("Validation Accuracy (%)", fontsize=22, labelpad=15)
plt.ylim(-5, 105)  # Give a little padding
plt.xlim(0, num_steps)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle="--", linewidth=1.5)  # Make grid lines bolder
plt.title("Hypothetical Model Performance on Associative Recall", fontsize=24, pad=20)

# Customize legend - place outside plot, increase font size
plt.legend(
    bbox_to_anchor=(1.04, 0.5),  # Position legend outside on the right
    loc="center left",
    borderaxespad=0,
    fontsize=20,  # Increase legend font size
    frameon=True,  # Add frame around legend
    edgecolor="black",
)

# Adjust layout to prevent labels/legend from being cut off
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space on the right for legend

plt.savefig("synthetic_results_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print("Synthetic results plot saved as synthetic_results_plot.png")
