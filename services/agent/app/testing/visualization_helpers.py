"""
Helper functions for visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List


def _get_difficulty_colormap():
    """
    Create custom colormap for difficulty: orange (easy) -> blue (medium) -> reddish purple (hard).
    :return: Colormap instance.
    """
    colors = [
        (1.0, 0.5, 0.0),      # Orange (easy - high average score)
        (0.0, 0.4, 0.8),      # Blue (medium)
        (0.6, 0.2, 0.5),      # Reddish purple (hard - low average score)
    ]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list("difficulty", colors, N=n_bins)
    return cmap


def _system_label(result: Dict) -> str:
    """
    Build label for execution system (model + variant).
    :param result: Test result.
    :return: System label.
    """
    model = str(result.get("model", "unknown"))
    variant = str(result.get("execution_variant", "graph"))
    return f"{model} [{variant}]"


def _format_tokens(value: float) -> str:
    """
    Format token count with k abbreviation for thousands.
    :param value: Token count.
    :return: Formatted string.
    """
    if value >= 1000:
        return f"{value/1000:.1f}k"
    return f"{int(value)}"


def _get_system_colors(models: List[str], colormap_name: str = "Set3") -> Dict[str, tuple]:
    """
    Assign colors to systems, making sequential variants darker than graph variants.
    :param models: List of system labels (model + variant).
    :param colormap_name: Name of matplotlib colormap to use.
    :return: Dictionary mapping system label to RGB color tuple.
    """
    base_model_names = {}
    model_to_base = {}
    
    for model in models:
        if "[sequential]" in model:
            base_name = model.split("[")[0].strip()
            base_model_names[base_name] = None
            model_to_base[model] = base_name
        elif "[graph]" in model:
            base_name = model.split("[")[0].strip()
            base_model_names[base_name] = None
            model_to_base[model] = base_name
        else:
            base_model_names[model] = None
            model_to_base[model] = model
    
    unique_bases = sorted(set(model_to_base.values()))
    colormap = plt.cm.get_cmap(colormap_name)
    base_colors = {base: colormap(i / len(unique_bases)) for i, base in enumerate(unique_bases)}
    
    system_colors = {}
    for model in models:
        base_name = model_to_base[model]
        base_color = base_colors[base_name]
        
        if "[sequential]" in model:
            rgb = np.array(base_color[:3])
            darkened = rgb * 0.4
            system_colors[model] = tuple(darkened) + (base_color[3],)
        else:
            system_colors[model] = base_color
    
    return system_colors
