import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# --- 色判定ロジック ---
def has_color(roi):
    if roi.size == 0: return False
    h, w = roi.shape[:2]
    # 中心部分を抽出して判定
    cp = roi[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    if cp.size == 0: return False
    avg_bgr = np.mean(cp, axis=(0, 1))
    
    # 1. 無彩色（白・黒・グレーすべて）を除外
    # 明るさに関係なく、色の偏り（鮮やかさ）が40未満なら「色なし」とみなす
    if (np.max(avg_bgr) - np.min(avg_bgr) < 40):
        return False
    return True

# --- 解析の実行核心部 ---
def run_analysis_core(img, presets, days=7):
    """
    days: 解析する日数（メインUIから渡される）
    """
    color_names = ["赤", "黄", "緑", "水色", "青", "紫", "ピンク"]
    all_results = []
    img_review = img.copy()
    h_img, w_img = img.shape[:2]

    for p in presets:
        # 比率から現在の画像の座標を復元
        x1 = int(p["rect"][0] * w_img)
        y1 = int(p["rect"][1] * h_img)
        x2 = int(p["rect"][2] * w_img)
        y2 = int(p["rect"][3] * h_img)
        
        s_num, e_num = p["start_num"], p["end_num"]
        num_units = abs(e_num - s_num) + 1
        step = 1 if e_num >= s_num else -1

        for i in range(num_units):
            unit_num = s_num + (i * step)
            row = {"台番号": unit_num}
            appearance_count = 0 
            
            # y座標（台の高さ方向）の計算
            unit_y = y1 + (y2 - y1) * i / (num_units - 1) if num_units > 1 else y1
            
            # x座標（日数方向）の計算を動的に行う
            for day in range(days):
                # days=1 の場合のゼロ除算を防ぐ
                if days > 1:
                    curr_x = int(x1 + (x2 - x1) * day / (days - 1))
                else:
                    curr_x = int(x1)
                
                curr_y = int(unit_y)
                
                # サンプリング範囲を抽出
                roi = img[max(0, curr_y-5):curr_y+5, max(0, curr_x-5):curr_x+5]
                
                if has_color(roi):
                    if appearance_count < len(color_names):
                        row[f"{day+1}日目"] = color_names[appearance_count]
                    else:
                        row[f"{day+1}日目"] = "他"
                    appearance_count += 1
                    # ヒットした場所を緑の丸で表示
                    cv2.circle(img_review, (curr_x, curr_y), 5, (0, 255, 0), -1)
                else:
                    row[f"{day+1}日目"] = ""
                    # ヒットしなかった場所を赤の丸で表示
                    cv2.circle(img_review, (curr_x, curr_y), 5, (0, 0, 255), -1)
            
            all_results.append(row)
    
    return all_results, img_review

# logic.py の末尾に追加

def find_best_match(ocr_list, master_db):
    """
    OCR結果のリストとDB内のマスターリストを比較し、最も一致率が高いものを返す。
    """
    row_count = len(ocr_list)
    key = f"{row_count}_rows"
    
    # 同じ台数のデータがDBになければ終了
    if key not in master_db or not master_db[key]:
        return None, 0

    best_master = None
    max_score = 0

    for entry in master_db[key]:
        master_list = entry["list"]
        match_count = 0
        
        # 各要素を比較して一致数をカウント (インデックスがズレない前提)
        # 台数が同じなので zip で回せる
        for ocr_val, master_val in zip(ocr_list, master_list):
            if str(ocr_val).strip() == str(master_val).strip():
                match_count += 1
        
        score = match_count / row_count
        if score > max_score:
            max_score = score
            best_master = entry

    return best_master, max_score