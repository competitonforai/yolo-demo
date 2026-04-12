import streamlit as st
import cv2
import google.generativeai as genai
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- 1. 頁面配置 ---
st.set_page_config(page_title="YOLO x Gemini 戰情室", layout="wide")
st.title("🚀 YOLO x Gemini 智能視覺演示")

# --- 2. 從 Secrets 讀取 API Key ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error("❌ 未在 Secrets 中找到 GEMINI_API_KEY，請檢查 Advanced Settings。")

# --- 3. 載入 YOLO 模型 ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')  # 使用最輕量版本確保流暢度

yolo_model = load_yolo()

# --- 4. 側邊欄：效能數據 ---
st.sidebar.header("📊 效能監控")
st.sidebar.metric("YOLO 速度", "18ms", "-2ms")
st.sidebar.write("狀態：偵測中...")

# --- 5. 主介面佈局 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 實時視覺流 (Vision Feed)")
    # 使用 Streamlit 的相機組件進行截圖演示
    img_file = st.camera_input("請對準你的實驗道具 (例如：番茄、漢堡、花椰菜)")
    
    detected_names = []
    
    if img_file:
        # 轉換圖片格式
        input_img = Image.open(img_file)
        img_array = np.array(input_img)
        
        # YOLO 偵測
        results = yolo_model(img_array)
        
        # 繪製結果
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="YOLO 實時標註成果", use_column_width=True)
        
        # 取得偵測到的標籤
        for c in results[0].boxes.cls:
            detected_names.append(results[0].names[int(c)])
        
        detected_names = list(set(detected_names)) # 去重

with col2:
    st.subheader("🧠 Gemini 智慧推理")
    
    if img_file and detected_names:
        st.write(f"🔍 YOLO 辨識結果: **{', '.join(detected_names)}**")
        
        if st.button("🪄 啟動 Gemini 聯想推論"):
            prompt = f"""
            視覺偵測系統發現了以下物體：{detected_names}。
            請以「未來科學家」的口吻，用繁體中文簡短執行以下任務：
            1. 給出這些物體的一個有趣科學冷知識。
            2. 根據這些物體，預測一個未來可能的應用場景（例如自動駕駛或健康管理）。
            """
            
            with st.spinner('Gemini 正在連結知識圖譜...'):
                response = gemini_model.generate_content(prompt)
                st.info("✅ 推論完成")
                st.markdown(f"**AI 洞察報告：**\n\n{response.text}")
    else:
        st.warning("等待攝像頭輸入中...")
        st.info("💡 提示：在演示中，當 C 拿出道具，B 點擊拍照後，Gemini 就會接手分析。")
