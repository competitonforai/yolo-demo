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
    # 這裡會讀取你在 Streamlit Advanced Settings 填寫的內容
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    # 使用最穩定的模型名稱
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error("❌ 未在 Secrets 中找到 GEMINI_API_KEY，或是格式錯誤。請檢查 Advanced Settings。")

# --- 3. 載入 YOLO 模型 ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')  # 使用最輕量版本確保流暢度

yolo_model = load_yolo()

# --- 4. 側邊欄：效能數據 ---
st.sidebar.header("📊 效能監控")
st.sidebar.metric("YOLO 速度", "18ms", "-2ms")
st.sidebar.write("狀態：準備就緒")

# --- 5. 主介面佈局 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 實時視覺流 (Vision Feed)")
    
    # 【修改處】：這裡從相機輸入改成了檔案上傳
    img_file = st.file_uploader("請上傳實驗照片 (jpg, png, jpeg)", type=['jpg', 'png', 'jpeg'])
    
    detected_names = []
    
    if img_file:
        # 1. 讀取上傳的檔案並轉為圖片 (這是你問的那行！)
        input_img = Image.open(img_file)
        img_array = np.array(input_img)
        
        # 2. 顯示給觀眾看
        st.image(input_img, caption="已上傳的照片", use_container_width=True)
        
        # 3. 執行 YOLO 偵測
        results = yolo_model(img_array)
        
        # 4. 繪製偵測結果框框
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="YOLO 實時標註成果", use_container_width=True)
        
        # 5. 取得偵測到的標籤
        for c in results[0].boxes.cls:
            detected_names.append(results[0].names[int(c)])
        
        detected_names = list(set(detected_names)) # 去除重複名稱

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
                try:
                    response = gemini_model.generate_content(prompt)
                    st.info("✅ 推論完成")
                    st.markdown(f"**AI 洞察報告：**\n\n{response.text}")
                except Exception as e:
                    st.error(f"Gemini 呼叫失敗，請檢查 API Key 或模型名稱：{e}")
    else:
        st.warning("等待照片上傳中...")
        st.info("💡 提示：請先在左側上傳實驗照片，當 YOLO 辨識出物體後，按按鈕叫 Gemini 分析。")
