import matplotlib.pyplot as plt
import seaborn as sns


# Helper function to visualize palettes
def plot_palette(palette, title="Color Palette"):
    """Helper function to plot a seaborn color palette"""
    print(f"\n--- {title} ---")
    print(palette)
    sns.palplot(palette)
    plt.title(title)
    # plt.show() # Uncomment to display during direct execution


# --- Palette Definitions ---

# 1. Custom 'Vibrant Thesis' Palette (Recommended)
# Manually curated for high impact and distinction. Matches example order.
# Note: Colors assigned based on visual matching to the example plot.
# DeltaNet (Red), Mamba (Green), GLA (Blue), RetNet (Cyan), RWKV (Purple), Hyena (Orange)
palette_vibrant_thesis = [
    "#FF0000",  # Bold Red (Matching DeltaNet) - Adjusted for pure red
    "#66CC00",  # Vibrant Green (Matching Mamba) - Adjusted slightly
    "#3399FF",  # Strong Sky Blue (Matching GLA) - Adjusted slightly
    "#1EEBEB",  # Bright Cyan (Matching RetNet) - Adjusted slightly
    "#9933FF",  # Vivid Purple (Matching RWKV) - Adjusted slightly
    "#FFAA00",  # Bright Orange (Matching Hyena) - Adjusted slightly
]

# 2. Seaborn 'bright' Palette
# Seaborn has a built-in 'bright' theme which is a good starting point.
palette_sns_bright = sns.color_palette("bright", 6)  # Get 6 colors

# 3. Custom 'Bold Neon' Palette
# Pushing vibrancy further, potentially bordering on neon. Use carefully.
palette_bold_neon = [
    "#FF1A66",  # Hot Pink/Red
    "#33FF1A",  # Lime Green (Swapped order)
    "#1A8CFF",  # Electric Blue (Swapped order)
    "#33FFFF",  # Electric Cyan
    "#A633FF",  # Bright Violet (Swapped order)
    "#FF9933",  # Electric Orange (Swapped order)
]

# 4. Seaborn HUSL-Generated Vibrant Palette
# Use HUSL color space for well-separated hues with high saturation.
palette_husl_vibrant = sns.husl_palette(n_colors=6, s=0.9, l=0.55)  # Use husl_palette directly (s/l are 0-1)

# Define the main palette to be easily imported
selected_palette = palette_vibrant_thesis

# Example of viewing palettes if script is run directly
if __name__ == "__main__":
    print("Visualizing Palettes defined in palette.py")
    plot_palette(palette_vibrant_thesis, "Custom 'Vibrant Thesis' Palette")
    plot_palette(palette_sns_bright, "Seaborn 'bright' Palette (6 Colors)")
    plot_palette(palette_bold_neon, "Custom 'Bold Neon' Palette")
    plot_palette(palette_husl_vibrant, "Seaborn HUSL Vibrant Palette (s=90, l=55)")
    plt.show()  # Show all palettes at the end
