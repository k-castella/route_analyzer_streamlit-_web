import cv2
import numpy as np
import pandas as pd
import sys
import streamlit as st
from collections import Counter

# --- 判定部品のみlogic.pyから拝借 ---
try:
    from logic import has_color
except ImportError:
    print("DEBUG: logic.py が見つかりません。")
    sys.exit()

# ★ OCR関連の関数（get_ocr_reader, perform_ocr_on_cell）は完全に排除しました ★

# --- 既存の検出ロジック (変更なし) ---

def auto_detect_all_tables(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = cv2.split(hsv)[2]
    _, mask = cv2.threshold(v, 160, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thick_lines = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_rects = []
    img_area = img.shape[0] * img.shape[1]
    for cnt in contours:
        if cv2.contourArea(cnt) > (img_area * 0.01):
            x, y, w, h = cv2.boundingRect(cnt)
            found_rects.append((x, y, w, h))
    found_rects.sort(key=lambda r: (r[1] // 50, r[0]))
    return found_rects

def group_lines(points):
    if len(points) == 0: return []
    groups = []; points.sort(); current_group = [points[0]]
    for i in range(1, len(points)):
        if points[i] - points[i-1] < 8: current_group.append(points[i])
        else:
            groups.append(int(np.mean(current_group))); current_group = [points[i]]
    groups.append(int(np.mean(current_group)))
    return groups

def filter_by_spacing(points):
    if len(points) < 2: return points
    spacings = [points[i] - points[i-1] for i in range(1, len(points))]
    median_spacing = np.median(spacings)
    filtered = [points[0]]
    for i in range(1, len(points)):
        if (points[i] - filtered[-1]) > (median_spacing * 0.7):
            filtered.append(points[i])
    return filtered

def find_grid_in_table(img, x, y, w, h):
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 25)
    v_density = np.sum(binary, axis=0)
    detected_x = group_lines(np.where(v_density > (h * 255 * 0.35))[0])
    h_density = np.sum(binary, axis=1)
    y_points = np.where(h_density > (w * 255 * 0.3))[0]
    final_y = filter_by_spacing(group_lines(y_points))

    if len(detected_x) >= 3:
        right_side_spacings = [detected_x[i] - detected_x[i-1] for i in range(len(detected_x)-5, len(detected_x)) if i > 0]
        std_w = np.median(right_side_spacings)
    else: std_w = w / 16

    temp_x = []
    curr = w
    while curr > (std_w * 1.2):
        temp_x.append(curr)
        curr -= int(std_w)
    
    return sorted([0] + temp_x), sorted(final_y)

# --- 解析実行部 ---

def execute_auto_analysis_full(img, skip_ocr_with_list=None, expected_days=None, expected_rows=None):
    tables = auto_detect_all_tables(img)
    all_results = []
    img_review = img.copy()
    color_names = ["赤", "黄", "緑", "水色", "青", "紫", "ピンク"]

    current_dai_count = 1  # OCRの代わりの連番カウンター
    current_list_idx = 0 
    last_rx_list = []

    progress_text = "解析中..."
    my_bar = st.progress(0, text=progress_text)
    total_tables = len(tables)

    for t_idx, (tx, ty, tw, th) in enumerate(tables):
        progress_val = int((t_idx / total_tables) * 100)
        my_bar.progress(progress_val, text=f"テーブル {t_idx+1}/{total_tables} を解析中...")

        rx_list, ry_list = find_grid_in_table(img, tx, ty, tw, th)
        if len(rx_list) < 2 or len(ry_list) < 3: continue
        last_rx_list = rx_list

        table_dais = []
        for r_idx in range(1, len(ry_list) - 1):
            # 外部リストがある場合はそちらを優先、ない場合は1からの連番を振る
            if skip_ocr_with_list is not None:
                if current_list_idx < len(skip_ocr_with_list):
                    val = skip_ocr_with_list[current_list_idx]
                else:
                    val = f"未定義_{current_list_idx+1}"
                current_list_idx += 1
            else:
                # ★ ここが重要：OCRを行わず連番を割り当てる ★
                val = str(current_dai_count)
                current_dai_count += 1
                
            table_dais.append(val)
        
        # analyze_dai_sequence(OCR補正)は不要になったため削除しました

        for r_idx in range(1, len(ry_list) - 1):
            # 台番号をセット（補正フラグ等も不要なのでシンプルに）
            dai_name = table_dais[r_idx-1]
            row_data = {"台番号": dai_name}
            
            appearance_count = 0 
            y_mid = int(ty + (ry_list[r_idx] + ry_list[r_idx+1]) / 2.0)
            
            for c_idx in range(1, len(rx_list) - 1):
                x_mid = int(tx + (rx_list[c_idx] + rx_list[c_idx+1]) / 2.0)
                roi = img[max(0, y_mid-5):y_mid+5, max(0, x_mid-5):x_mid+5]
                if has_color(roi):
                    c_name = color_names[appearance_count] if appearance_count < len(color_names) else "他"
                    row_data[f"{c_idx}日目"] = c_name
                    appearance_count += 1
                    cv2.circle(img_review, (x_mid, y_mid), 4, (0, 255, 0), -1)
                else:
                    row_data[f"{c_idx}日目"] = ""
                    cv2.circle(img_review, (x_mid, y_mid), 4, (0, 0, 255), -1)
            
            # --- ハイライト処理 ---
            has_hit = any(v in color_names for k, v in row_data.items() if k != "台番号")
            y1_h, y2_h = ty + ry_list[r_idx], ty + ry_list[r_idx+1]

            if has_hit:
                # 当たりあり：行を暗くする
                roi = img_review[y1_h:y2_h, tx:tx+tw]
                img_review[y1_h:y2_h, tx:tx+tw] = (roi.astype(np.float32) * 0.80).astype(np.uint8)
            else:
                # 当たりなし：そのまま（必要ならここで色を乗せる）
                pass
            
            all_results.append(row_data)

    my_bar.empty()

    actual_rows = len(all_results)
    actual_days = len(last_rx_list) - 2 if last_rx_list else 0
    
    # プリセット不一致警告表示（ロジック維持）
    row_err = (expected_rows is not None and expected_rows > 0 and expected_rows != actual_rows)
    day_err = (expected_days is not None and expected_days > 0 and expected_days != actual_days)
    if row_err or day_err:
        msg = "!! PRESET MISMATCH !!"
        details = f"Rows: {actual_rows}(exp:{expected_rows}) Days: {actual_days}(exp:{expected_days})"
        cv2.rectangle(img_review, (10, 10), (750, 140), (0, 0, 0), -1)
        cv2.putText(img_review, msg, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        cv2.putText(img_review, details, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return all_results, img_review