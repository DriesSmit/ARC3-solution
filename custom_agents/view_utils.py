import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import os

# Color mapping for ARC grid visualization
key_colors = {
    0: "#FFFFFF",   # White
    1: "#CCCCCC",   # Light Gray
    2: "#999999",   # Gray
    3: "#666666",   # Dark Gray
    4: "#333333",   # Darker Gray
    5: "#000000",   # Black
    6: "#E53AA3",   # Pink
    7: "#FF7BCC",   # Light Pink
    8: "#F93C31",   # Red
    9: "#1E93FF",   # Blue
    10: "#88D8F1",  # Light Blue
    11: "#FFDC00",  # Yellow
    12: "#FF851B",  # Orange
    13: "#921231",  # Dark Red
    14: "#4FCC30",  # Green
    15: "#A356D6"   # Purple
}

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_grid_image(grid: np.ndarray, cell_size: int = 8, border_width: int = 1) -> Image.Image:
    """Create a PIL image from an ARC grid."""
    # Ensure grid is 2D
    if len(grid.shape) == 3 and grid.shape[0] == 1:
        grid = grid.squeeze(0)
    
    height, width = grid.shape
    img_width = width * cell_size + (width + 1) * border_width
    img_height = height * cell_size + (height + 1) * border_width
    
    # Create image with white background
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid cells
    for y in range(height):
        for x in range(width):
            color_idx = int(grid[y, x])
            color = hex_to_rgb(key_colors[color_idx])
            
            # Calculate cell position
            left = x * (cell_size + border_width) + border_width
            top = y * (cell_size + border_width) + border_width
            right = left + cell_size
            bottom = top + cell_size
            
            # Draw cell
            draw.rectangle([left, top, right, bottom], fill=color)
    
    return img

def create_transition_image(before_grid: np.ndarray, after_grid: np.ndarray, 
                          action_info: str = "", cell_size: int = 8) -> Image.Image:
    """Create a transition image showing before and after grids side by side."""
    before_img = create_grid_image(before_grid, cell_size)
    after_img = create_grid_image(after_grid, cell_size)
    
    # Create combined image
    padding = 20
    text_height = 30
    img_width = before_img.width + after_img.width + padding * 3
    img_height = max(before_img.height, after_img.height) + text_height * 3
    
    combined_img = Image.new('RGB', (img_width, img_height), 'white')
    
    # Paste images
    combined_img.paste(before_img, (padding, text_height * 2))
    combined_img.paste(after_img, (before_img.width + padding * 2, text_height * 2))
    
    # Add labels
    draw = ImageDraw.Draw(combined_img)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add "Before" and "After" labels
    draw.text((padding + before_img.width // 2 - 20, text_height), "Before", 
              fill='black', font=font)
    draw.text((before_img.width + padding * 2 + after_img.width // 2 - 15, text_height), 
              "After", fill='black', font=font)
    
    # Add action info
    if action_info:
        # Truncate long action info
        if len(action_info) > 80:
            action_info = action_info[:77] + "..."
        draw.text((padding, 5), f"Action: {action_info}", fill='black', font=font)
    
    return combined_img

def save_random_transitions(frames_history: list, action_history: list, 
                          log_dir: str, step: int, num_samples: int = 5) -> None:
    """Save random transition images from recent history."""
    if len(frames_history) < 2 or len(action_history) < 1:
        return
    
    # Create transitions directory
    transitions_dir = os.path.join(log_dir, "transitions")
    os.makedirs(transitions_dir, exist_ok=True)
    
    # Get available transitions (need at least 2 frames for a transition)
    max_transitions = min(len(frames_history) - 1, len(action_history))
    if max_transitions < 1:
        return
    
    # Sample random transitions
    num_samples = min(num_samples, max_transitions)
    transition_indices = np.random.choice(max_transitions, size=num_samples, replace=False)
    
    for i, idx in enumerate(transition_indices):
        before_frame = frames_history[idx]
        after_frame = frames_history[idx + 1]
        action = action_history[idx]
        
        # Convert frames to numpy arrays
        before_grid = np.array(before_frame.frame)
        after_grid = np.array(after_frame.frame)
        
        # Take the last frame (in case of animation)
        if len(before_grid.shape) == 3:
            before_grid = before_grid[-1]
        if len(after_grid.shape) == 3:
            after_grid = after_grid[-1]
        
        # Create action info string
        if hasattr(action, 'reasoning'):
            if isinstance(action.reasoning, dict):
                action_info = f"{action.value} at {action.reasoning.get('coordinates', 'N/A')}"
            else:
                action_info = f"{action.value}: {str(action.reasoning)[:50]}"
        else:
            action_info = str(action.value)
        
        # Create and save transition image
        filename = f"transition_step{step}_sample{i+1}.png"
        save_path = os.path.join(transitions_dir, filename)
        
        transition_img = create_transition_image(before_grid, after_grid, action_info)
        transition_img.save(save_path)

def create_action_prob_chart(action_probs: np.ndarray, selected_action_idx: int, 
                           chart_width: int = 400, chart_height: int = 120) -> Image.Image:
    """Create a bar chart showing action probabilities."""
    action_names = ["Up", "Down", "Left", "Right", "Space", "Click"]
    
    # Create image with white background
    img = Image.new('RGB', (chart_width, chart_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Chart parameters
    margin = 20
    bar_width = (chart_width - 2 * margin) // len(action_probs)
    max_bar_height = chart_height - 40
    
    # Draw bars
    for i, prob in enumerate(action_probs):
        x = margin + i * bar_width
        bar_height = int(prob * max_bar_height)
        y = chart_height - 20 - bar_height
        
        # Color: green for selected action, blue for others
        color = (0, 200, 0) if i == selected_action_idx else (100, 150, 255)
        
        # Draw bar
        draw.rectangle([x + 2, y, x + bar_width - 2, chart_height - 20], fill=color)
        
        # Draw probability text
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        prob_text = f"{prob:.3f}"
        draw.text((x + 5, y - 15), prob_text, fill='black', font=font)
        
        # Draw action name
        action_name = action_names[i] if i < len(action_names) else f"A{i}"
        draw.text((x + 2, chart_height - 15), action_name, fill='black', font=font)
    
    # Draw title
    draw.text((10, 5), "Action Probabilities", fill='black', font=font)
    
    return img

def create_click_prob_visualization(grid: np.ndarray, click_probs: np.ndarray, 
                                  selected_click_idx: int, cell_size: int = 8) -> Image.Image:
    """Visualize click probabilities as a heatmap overlay on grid (64x64 array input)."""
    # Create base grid image
    grid_img = create_grid_image(grid, cell_size)
    
    # Create overlay
    overlay = Image.new('RGBA', grid_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    grid_height, grid_width = grid.shape
    border_width = 1
    
    # Get probability stats for visualization
    max_prob = click_probs.max()
    
    # Draw heatmap for all cells using raw probabilities (no normalization)
    for y in range(min(grid_height, 64)):
        for x in range(min(grid_width, 64)):
            prob = click_probs[y, x]  # Raw probability 0-1
            
            # Calculate pixel position for cell center (small dots, not full rectangles)
            center_x = x * (cell_size + border_width) + border_width + cell_size // 2
            center_y = y * (cell_size + border_width) + border_width + cell_size // 2
            
            # Transparency directly from raw probability
            alpha = int(prob * 100)  # Scale raw prob to reasonable alpha (max 100)
            
            # Heat map colors: blue (low) to yellow to red (high)
            if prob < 0.5:
                # Blue to yellow
                red = int(255 * (prob * 2))
                green = int(255 * (prob * 2))
                blue = 255 - int(128 * (prob * 2))
            else:
                # Yellow to red
                red = 255
                green = 255 - int(255 * ((prob - 0.5) * 2))
                blue = 0
            
            color = (red, green, blue, alpha)
            
            # Draw small dots instead of full rectangles
            dot_size = 2
            draw.ellipse([center_x - dot_size, center_y - dot_size, 
                         center_x + dot_size, center_y + dot_size], fill=color)
    
    # Highlight selected position if valid
    if selected_click_idx >= 0:
        sel_y = selected_click_idx // 64
        sel_x = selected_click_idx % 64
        
        if sel_x < grid_width and sel_y < grid_height:
            left = sel_x * (cell_size + border_width) + border_width
            top = sel_y * (cell_size + border_width) + border_width
            right = left + cell_size
            bottom = top + cell_size
            
            draw.rectangle([left-1, top-1, right+1, bottom+1], outline=(255, 0, 0, 255), width=2)
    
    # Add info text
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    info_text = f"Click Heatmap (max: {max_prob:.3f}, raw 0-1 values)"
    draw.text((5, 5), info_text, fill=(255, 255, 255, 200), font=font)
    
    # Composite overlay onto grid
    result = Image.alpha_composite(grid_img.convert('RGBA'), overlay)
    return result.convert('RGB')

def save_action_visualization(frame: 'FrameData', action_probs: np.ndarray, 
                            click_probs: np.ndarray, selected_action_idx: int,
                            selected_click_idx: int, log_dir: str, step: int, sample_id: int = 1) -> None:
    """Save visualization showing current frame with action and click probabilities."""
    # Create output directory
    viz_dir = os.path.join(log_dir, "action_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert frame to numpy array
    grid = np.array(frame.frame)
    if len(grid.shape) == 3:
        grid = grid[-1]  # Take last frame if animation
    
    # Create action probability chart
    action_chart = create_action_prob_chart(action_probs, selected_action_idx)
    
    # Create frame visualization - always show heatmap if click_probs provided
    if click_probs is not None:
        frame_viz = create_click_prob_visualization(grid, click_probs, selected_click_idx)
        viz_type = "heatmap"
        # Debug info
        # print(f"Creating heatmap visualization: action_idx={selected_action_idx}, click_idx={selected_click_idx}, click_probs_shape={click_probs.shape}")
    else:
        frame_viz = create_grid_image(grid, cell_size=8)
        viz_type = "simple"
        # Debug info
        # print(f"Creating simple visualization: action_idx={selected_action_idx}, no click_probs provided")
    
    
    # Combine action chart and frame visualization
    padding = 10
    total_width = max(action_chart.width, frame_viz.width)
    total_height = action_chart.height + frame_viz.height + padding
    
    combined_img = Image.new('RGB', (total_width, total_height), 'white')
    
    # Center images
    action_x = (total_width - action_chart.width) // 2
    frame_x = (total_width - frame_viz.width) // 2
    
    combined_img.paste(action_chart, (action_x, 0))
    combined_img.paste(frame_viz, (frame_x, action_chart.height + padding))
    
    # Save image with action name  
    action_names = ["Up", "Down", "Left", "Right", "Space", "Click"]
    action_name = action_names[selected_action_idx] if selected_action_idx < len(action_names) else f"ACTION{selected_action_idx}"
    filename = f"step{step}_{action_name}.png"
    save_path = os.path.join(viz_dir, filename)
    combined_img.save(save_path)