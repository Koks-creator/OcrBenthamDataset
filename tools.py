"""
Module for automatically sorting detected text areas in natural reading order.
Detects document layout and selects the optimal strategy.

MAIN FEATURES:
---------------

1. smart_sort_text_lines(boxes, debug=False)
   - Strategy: MAIN COLUMN FIRST, then marginalia
   - Uses DBSCAN to detect columns based on X position
   - Identifies the main column (widest + most lines)
   - Sorting: main column (top→bottom), then side notes (top→bottom)
   - Ideal for: text with marginalia, side notes, historical documents

2. sort_reading_order_universal(boxes, y_tolerance=30, debug=False)
   - Strategy: LINE BY LINE (left→right, top→bottom)
   - Groups boxes into rows (Y), then sorts by X
   - Uses e.g. lexsort() for efficient sorting
   - Ideal for: two-column text, newspapers, articles, tables

3. auto_sort_boxes(boxes, strategy='auto', debug=False)
   - AUTOMATIC STRATEGY SELECTION
   - Analysis: checks the standard deviation of box widths
   - If large differences in widths → 'main_first' (marginalia)
   - If similar widths → 'reading_order' (even layout)
"""

import numpy as np
from sklearn.cluster import DBSCAN


def smart_sort_text_lines(boxes: list, debug=False):
    """
    Intelligent text line sorting with automatic layout detection.
    
    Main concept:
    1. Use DBSCAN clustering to detect vertical columns based on X positions
    2. Identify the MAIN column (widest + most content)
    3. Sort main column first (top to bottom), then side notes/marginalia
    
    This approach prioritizes reading the primary content flow before 
    secondary annotations, similar to how medieval manuscripts or 
    annotated documents are read.
    
    Supports:
    - Single-column text (simple top-to-bottom sort)
    - Text with marginalia on left or right side
    - Evenly spaced two-column layouts
    
    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        debug: Print column detection info
    
    Returns:
        List of boxes sorted: main column first, then marginalia
    """
    if len(boxes) == 0:
        return []
    
    boxes_arr = np.array(boxes)
    
    # Calculate geometric properties for each box
    x_centers = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2  # Horizontal center point
    y_centers = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2  # Vertical center point
    widths = boxes_arr[:, 2] - boxes_arr[:, 0]           # Box width
    
    # DBSCAN clustering to detect vertical columns based on X-axis position
    # eps=50: Boxes within 50 pixels horizontally are considered same column
    # min_samples=2: Need at least 2 boxes to form a column (noise otherwise)
    # This automatically finds columns without knowing their count in advance
    clustering = DBSCAN(eps=50, min_samples=2).fit(x_centers.reshape(-1, 1))
    labels = clustering.labels_  # Each box gets a column label (or -1 for noise)
    
    # Count detected columns (excluding noise labeled as -1)
    n_columns = len(set(labels)) - (1 if -1 in labels else 0)
    
    if debug:
        print(f"Detected columns: {n_columns}")
    
    # CASE 1: Single column detected - simple top-to-bottom sorting
    if n_columns <= 1:
        sorted_indices = np.argsort(y_centers)
        return [boxes[i] for i in sorted_indices]
    
    # CASE 2: Multiple columns - identify main column vs marginalia
    columns_info = []
    for label in set(labels):
        if label == -1:  # Skip noise (boxes not assigned to any column)
            continue
        
        # Extract all boxes belonging to this column
        mask = labels == label
        col_x_centers = x_centers[mask]
        col_widths = widths[mask]
        
        # Calculate column statistics
        avg_x = np.mean(col_x_centers)      # Average horizontal position
        avg_width = np.mean(col_widths)     # Average box width in column
        count = np.sum(mask)                # Number of boxes in column
        
        columns_info.append({
            'label': label,
            'avg_x': avg_x,
            'avg_width': avg_width,
            'count': count,
            'mask': mask
        })
    
    # Sort columns by horizontal position (left to right)
    columns_info.sort(key=lambda x: x['avg_x'])
    
    # Identify MAIN column using heuristic: width × count
    # Main column typically has: wider boxes AND more text lines
    # This product captures both aspects - distinguishes main text from narrow marginalia
    main_col_idx = max(range(len(columns_info)), 
                       key=lambda i: columns_info[i]['avg_width'] * columns_info[i]['count'])
    
    if debug:
        for i, col in enumerate(columns_info):
            marker = " <- MAIN" if i == main_col_idx else ""
            print(f"Col {i}: x={col['avg_x']:.0f}, width={col['avg_width']:.0f}, "
                  f"lines={col['count']}{marker}")
    
    # Build final sorted list: MAIN column first, then marginalia
    sorted_boxes = []
    
    # 1. Add main column boxes sorted top-to-bottom
    main_mask = columns_info[main_col_idx]['mask']
    main_indices = np.where(main_mask)[0]
    main_y = y_centers[main_indices]
    main_sorted = main_indices[np.argsort(main_y)]  # Sort by Y coordinate
    sorted_boxes.extend([boxes[i] for i in main_sorted])
    
    # 2. Add remaining columns (marginalia/side notes) sorted top-to-bottom
    for i, col_info in enumerate(columns_info):
        if i == main_col_idx:
            continue  # Skip main column (already added)
        
        mask = col_info['mask']
        indices = np.where(mask)[0]
        col_y = y_centers[indices]
        col_sorted = indices[np.argsort(col_y)]  # Sort by Y coordinate
        sorted_boxes.extend([boxes[i] for i in col_sorted])
    
    return sorted_boxes

def sort_reading_order_universal(boxes: list, y_tolerance: int = 30, debug: bool = False):
    """
    Sort bounding boxes in natural reading order: line by line, left to right.
    
    Main concept:
    1. Group boxes into horizontal rows (lines) based on Y coordinate
    2. Within each row, sort boxes from left to right (X coordinate)
    3. Process rows from top to bottom
    
    This mimics how humans read: finish one line completely before moving to next.
    Perfect for multi-column layouts, tables, and standard document formats.
    
    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        y_tolerance: Vertical grouping threshold in pixels (default: 30)
                    Boxes within y_tolerance pixels are considered same row.
                    Larger value = more aggressive row merging
                    Example: y_tolerance=30 means boxes at Y=100 and Y=125 
                            are treated as same line
        debug: Print row count for diagnostics
    
    Returns:
        List of boxes sorted in reading order
    """
    if len(boxes) == 0:
        return []
    
    boxes_arr = np.array(boxes)
    
    # Calculate center points for each box
    x_centers = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2
    y_centers = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2
    
    # Group boxes into rows (Y axis)
    # Divide Y by tolerance and round to create discrete row numbers
    # Example: y_tolerance=30 -> Y=95 and Y=110 both become row 3
    rows = np.round(y_centers / y_tolerance).astype(int)
    
    # Sort using lexicographic order: PRIMARY key = rows (Y), SECONDARY key = x_centers (X)
    # np.lexsort sorts by the LAST key first, then second-to-last, etc.
    # Result: boxes sorted by row first (top to bottom), then by X within each row (left to right)
    sort_indices = np.lexsort((x_centers, rows))
    
    if debug:
        print(f"Detected rows: {len(set(rows))}")
    
    return [boxes[i] for i in sort_indices]


# Wrapper - selects the best strategy
def auto_sort_boxes(boxes, strategy='auto', debug=False):
    """
    strategy: 'auto', 'main_first', 'reading_order'
    """
    if strategy == 'auto':
        # Try to figure out what is better
        boxes_arr = np.array(boxes)
        widths = boxes_arr[:, 2] - boxes_arr[:, 0]
        width_std = np.std(widths)
        
        # If there is a large difference in width -> probably marginalia
        if width_std > np.mean(widths) * 0.5:
            strategy = 'main_first'
        else:
            strategy = 'reading_order'
        
        if debug:
            print(f"Auto-selected strategy: {strategy}")
    
    if strategy == 'main_first':
        return smart_sort_text_lines(boxes, debug=debug)
    else:  # reading_order
        return sort_reading_order_universal(boxes, debug=debug)