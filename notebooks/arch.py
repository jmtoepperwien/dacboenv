import cv2
import numpy as np
import csv
import os
import math
from typing import Dict, List, Optional, Tuple, Any

# --- CONFIGURATION & CONSTANTS ---
SYMMETRY_THRESHOLD: float = 0.65 
FLASK_RIM_ANGLE: float = 45.0      
ANGLE_TOLERANCE: float = 15.0      
MIN_BOTTOM_RATIO: float = 0.10      # Discard bottom if < 10% of total vessel width
WALL_THRESHOLD_RATIO: float = 0.10  # Discard belly wall if > 10% of belly width
OUTPUT_DIR: str = 'detected'
PROFILE_DIR: str = 'profiles'
DIAG_DIR: str = 'diagnostics'
CSV_FILENAME: str = 'vessel_analysis.csv'

FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.5
FONT_THICKNESS: int = 1
COLORS: Dict[str, Tuple[int, int, int]] = {
    'axis': (180, 180, 180),
    'height': (0, 255, 0),
    'belly': (0, 255, 255),
    'rim': (255, 0, 255),               
    'bottom': (0, 165, 255),            # Orange
    'wall': (0, 0, 255),
    'profile_highlight': (0, 255, 0),
    'hull': (255, 255, 0),              
    'circle': (255, 0, 0)
}

def preprocess_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(image_path)
    if img is None: return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return img, gray, thresh

def find_central_axis(gray: np.ndarray, x_c: float) -> int:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=gray.shape[0]*0.3, maxLineGap=20)
    axis_x = int(x_c)
    if lines is not None:
        vertical_lines = [(l[0][0] + l[0][2]) / 2 for l in lines if abs(l[0][0] - l[0][2]) < 5]
        if vertical_lines:
            axis_x = int(min(vertical_lines, key=lambda val: abs(val - x_c)))
    return axis_x

def detect_vessel_class(hull_pts: np.ndarray, axis_x: int) -> Tuple[str, float]:
    left_points = np.sum(hull_pts[:, 0] < axis_x)
    right_points = np.sum(hull_pts[:, 0] > axis_x)
    total = left_points + right_points
    if total == 0: return "unknown", 0.0
    ratio = max(left_points, right_points) / total
    v_class = "pilgrims_flask" if ratio > SYMMETRY_THRESHOLD else "jar"
    return v_class, round(ratio, 2)

def get_flask_rim_segment(hull_pts: np.ndarray, axis_x: int) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    max_length = 0.0
    best_segment = None
    for i in range(len(hull_pts)):
        p1, p2 = hull_pts[i], hull_pts[(i + 1) % len(hull_pts)]
        if p1[0] < axis_x and p2[0] < axis_x:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle > 90: angle = 180 - angle
            if abs(angle - FLASK_RIM_ANGLE) <= ANGLE_TOLERANCE:
                length = math.sqrt(dx**2 + dy**2)
                if length > max_length:
                    max_length = length
                    best_segment = (p1, p2)
    return round(max_length, 2), best_segment

def calculate_bottom_stand_extended(thresh: np.ndarray, axis_x: int, total_width: int) -> Tuple[Optional[float], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    h, w = thresh.shape
    all_pts = np.argwhere(thresh == 255) 
    if len(all_pts) == 0: return None, None, None
    max_y = np.max(all_pts[:, 0])
    lowest_pixels = all_pts[all_pts[:, 0] == max_y]
    max_dist = 0
    for p in lowest_pixels:
        dist = abs(p[1] - axis_x)
        if dist > max_dist:
            max_dist = dist
    width = max_dist * 2
    if width < (total_width * MIN_BOTTOM_RATIO):
        return None, None, None
    p_left = (int(axis_x - max_dist), int(max_y))
    p_right = (int(axis_x + max_dist), int(max_y))
    return float(width), p_left, p_right

def extract_rightmost_outer_profile(thresh: np.ndarray, axis_x: int, filename: str, initial_diag: np.ndarray) -> np.ndarray:
    """Extracts profile starting from axis_x without a dead zone."""
    h, w = thresh.shape
    right_mask = np.zeros_like(thresh)
    right_mask[:, axis_x:] = 255
    right_side = cv2.bitwise_and(thresh, right_mask)
    
    # Cleaning vertical drawing artifacts
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_structures = cv2.morphologyEx(right_side, cv2.MORPH_OPEN, kernel_v)
    right_cleaned = cv2.subtract(right_side, vertical_structures)
    
    profile_mask = np.zeros_like(right_cleaned)
    for y in range(h):
        on_pixels = np.where(right_cleaned[y, :] == 255)[0]
        if len(on_pixels) > 0:
            profile_mask[y, on_pixels[-1]] = 255
            
    if not os.path.exists(PROFILE_DIR): os.makedirs(PROFILE_DIR)
    cv2.imwrite(os.path.join(PROFILE_DIR, f"profile_{filename}"), profile_mask)
    
    diag_base = cv2.cvtColor(right_side, cv2.COLOR_GRAY2BGR)
    diag_base = cv2.addWeighted(diag_base, 0.3, np.zeros_like(diag_base), 0.7, 0)
    profile_points = cv2.dilate(profile_mask, np.ones((3,3), np.uint8))
    diag_base[profile_points == 255] = COLORS['profile_highlight']
    dual_diag = np.hstack((initial_diag, diag_base))
    
    if not os.path.exists(DIAG_DIR): os.makedirs(DIAG_DIR)
    cv2.imwrite(os.path.join(DIAG_DIR, f"dual_diag_{filename}"), dual_diag)
    return dual_diag

def get_row_thickness_data(mask: np.ndarray, y_coord: int) -> Tuple[Optional[int], Optional[int], int]:
    if y_coord >= mask.shape[0]: return None, None, 0
    row = mask[y_coord, :]
    on_pixels = np.where(row == 255)[0]
    if len(on_pixels) >= 2:
        return int(on_pixels[0]), int(on_pixels[-1]), int(on_pixels[-1] - on_pixels[0])
    return None, None, 0

def annotate_render(draw_img: np.ndarray, m: Dict[str, Any]) -> None:
    cv2.line(draw_img, (m['axis_x'], 0), (m['axis_x'], draw_img.shape[0]), COLORS['axis'], 1)
    cv2.putText(draw_img, f"CLASS: {m['vessel_class'].upper()}", (10, 30), FONT, 0.7, (0, 0, 255), 2)
    h_x = int(m['max_x'] + 30)
    cv2.line(draw_img, (h_x, m['min_y']), (h_x, m['max_y']), COLORS['height'], 2)
    cv2.putText(draw_img, f"H: {m['height']}px", (h_x + 5, int(m['min_y'] + m['height']/2)), FONT, FONT_SCALE, COLORS['height'], FONT_THICKNESS)
    cv2.line(draw_img, (m['min_x'], m['belly_y']), (m['max_x'], m['belly_y']), COLORS['belly'], 2)
    cv2.putText(draw_img, f"Belly: {m['belly_width']}px", (m['min_x'], m['belly_y'] - 10), FONT, FONT_SCALE, COLORS['belly'], FONT_THICKNESS)
    
    if m['vessel_class'] == "pilgrims_flask" and m['rim_seg']:
        cv2.line(draw_img, tuple(m['rim_seg'][0]), tuple(m['rim_seg'][1]), COLORS['rim'], 3)
    else:
        rx1, rx2 = int(m['axis_x'] - m['opening_width']/2), int(m['axis_x'] + m['opening_width']/2)
        cv2.line(draw_img, (rx1, m['min_y']), (rx2, m['min_y']), COLORS['rim'], 3)
    
    if m['bottom_width'] is not None:
        cv2.line(draw_img, m['bottom_p1'], m['bottom_p2'], COLORS['bottom'], 3)
    
    wall_txt = f"Wall: {m['belly_wall']}px" if m['belly_wall'] is not None else "Wall: N/A"
    cv2.putText(draw_img, wall_txt, (10, 120), FONT, FONT_SCALE, COLORS['wall'], FONT_THICKNESS)

def analyze_vessel(image_path: str) -> Optional[Dict[str, Any]]:
    img_name = os.path.basename(image_path)
    img, gray, thresh = preprocess_image(image_path)
    if img is None or thresh is None: return None
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    clean_thresh = np.zeros_like(thresh)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 5: clean_thresh[labels == i] = 255
    
    points = cv2.findNonZero(clean_thresh)
    hull = cv2.convexHull(points); hull_pts = hull.reshape(-1, 2)
    axis_x = find_central_axis(gray, np.mean(hull_pts[:, 0]))
    
    min_x, max_x = int(np.min(hull_pts[:, 0])), int(np.max(hull_pts[:, 0]))
    min_y, max_y = int(np.min(hull_pts[:, 1])), int(np.max(hull_pts[:, 1]))
    total_width, height = max_x - min_x, max_y - min_y
    
    bottom_width, b_p1, b_p2 = calculate_bottom_stand_extended(clean_thresh, axis_x, total_width)
    v_class, _ = detect_vessel_class(hull_pts, axis_x)
    
    opening_width = 0.0
    rim_seg = None
    if v_class == "pilgrims_flask":
        opening_width, rim_seg = get_flask_rim_segment(hull_pts, axis_x)
    else:
        top_pts = hull_pts[hull_pts[:, 1] <= min_y + (height * 0.05)]
        opening_width = float((max([abs(p[0] - axis_x) for p in top_pts]) * 2) if len(top_pts) > 0 else 0)

    # Wall Thickness
    belly_y = int(hull_pts[np.argmax(hull_pts[:, 0]), 1])
    left_mask = np.zeros_like(clean_thresh); left_mask[:, :axis_x] = 255
    left_only = cv2.bitwise_and(clean_thresh, left_mask)
    wall_only = cv2.subtract(left_only, cv2.morphologyEx(left_only, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))))
    _, _, b_thick = get_row_thickness_data(wall_only, belly_y)
    
    if b_thick > (total_width * WALL_THRESHOLD_RATIO):
        b_thick = None

    initial_diag = cv2.cvtColor(clean_thresh, cv2.COLOR_GRAY2BGR)
    initial_diag = cv2.addWeighted(initial_diag, 0.4, np.zeros_like(initial_diag), 0.6, 0)
    
    # Extract profile with NO DEAD ZONE
    dual_view = extract_rightmost_outer_profile(clean_thresh, axis_x, img_name, initial_diag)
    cv2.imshow("Diagnostics", dual_view)

    m_data = {
        'axis_x': axis_x, 'min_y': min_y, 'max_y': max_y, 'min_x': min_x, 'max_x': max_x,
        'belly_y': belly_y, 'height': height, 'belly_width': total_width,
        'opening_width': opening_width, 'rim_seg': rim_seg, 'bottom_width': bottom_width,
        'bottom_p1': b_p1, 'bottom_p2': b_p2, 'belly_wall': b_thick, 'vessel_class': v_class
    }

    draw_img = img.copy()
    annotate_render(draw_img, m_data)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"vessel_{img_name}"), draw_img)
    cv2.imshow("Final Analysis", draw_img)
    cv2.waitKey(1) 

    return {
        "filename": img_name, "vessel_class": v_class, "height_px": height, 
        "belly_width_px": total_width, "opening_width_px": opening_width, 
        "bottom_width_px": bottom_width, "belly_wall_px": b_thick
    }

def batch_process(folder_path: str) -> None:
    if not os.path.exists(folder_path): return
    all_data = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        data = analyze_vessel(os.path.join(folder_path, f))
        if data: all_data.append(data)
    if all_data:
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader(); writer.writerows(all_data)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    batch_process("input")