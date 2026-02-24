# =================================================================
# 【Web版 解析ツール 起動方法】
# 
# 1. ターミナル(VS CodeのTerminalなど)でこのファイルがあるフォルダへ移動
# 2. 以下のコマンドを実行:
#    streamlit run main.py
#
# ※ 'python main.py' では起動しないので注意。
# ※ ライブラリ未インストールの場合は: pip install streamlit easyocr pandas opencv-python
# =================================================================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image  # これを追加
import main_auto_ocr

st.set_page_config(page_title="Route Analyzer", layout="centered")

uploaded_file = st.file_uploader("画像を選択", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, 1)

    # 全自動解析の実行
    results, img_review = main_auto_ocr.execute_auto_analysis_full(img_orig)

    if results:
        # --- ここで確実に画像に変換 ---
        st.write("### 解析範囲の確認")
        try:
            # OpenCV(BGR) -> RGBに変換
            img_rgb = cv2.cvtColor(img_review, cv2.COLOR_BGR2RGB)
            # PILの画像オブジェクトに変換
            preview_img = Image.fromarray(img_rgb)
            # 表示 (use_column_widthに変更)
            st.image(preview_img, caption="解析ドット確認", width="stretch")
        except Exception as e:
            st.error(f"画像表示エラー: {e}")

        # --- CSVデータの整形 ---
        df = pd.DataFrame(results)
        if "台番号" in df.columns:
            df["台番号"] = df["台番号"].astype(str).str.replace(r"\(.*?\)", "", regex=True).str.strip()
        
        csv_data = df.to_csv(index=False, encoding="utf-8-sig")

        # --- 結果の表示 ---
        st.write("### 解析結果 (CSV)")
        st.text_area("表示", value=csv_data, height=200)
        st.download_button("CSVファイルをダウンロード", csv_data, "result.csv", "text/csv")
        
    else:
        st.error("解析に失敗しました。")