import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 1. 頁面與模型設定
st.set_page_config(page_title="AI 視覺戰情室", layout="wide")
st.title("🔬 AI 智能視覺：YOLO + Gemini 雙腦協作")

# 讀取 Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # 使用 Flash 版本：兼顧速度與視覺辨識力
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error("Secrets 設定錯誤，請確認 GEMINI_API_KEY。")

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt') # 保持 Nano 版確保 Streamlit 跑得動

yolo_model = load_yolo()

# 2. 介面設計
img_file = st.file_uploader("📤 上傳實驗照片進行深度分析", type=['jpg', 'png', 'jpeg'])

if img_file:
    col1, col2 = st.columns([1, 1])
    input_img = Image.open(img_file)

    with col1:
        st.subheader("🤖 YOLO 即時標註")
        # 執行 YOLO (設定信心門檻為 0.2 提高敏感度)
        results = yolo_model(input_img, conf=0.2)
        res_plotted = results[0].plot()
        st.image(res_plotted, use_container_width=True)
        
        # 取得標籤
        labels = [results[0].names[int(c)] for c in results[0].boxes.cls]
        st.write(f"初步辨識清單: {', '.join(labels) if labels else '未發現預設物體'}")

    with col2:
        st.subheader("🧠 Gemini 深度科學分析")
        if st.button("🪄 啟動 AI 視覺推論"):
            with st.spinner('AI 正在觀察圖片細節...'):
                # --- 關鍵修正：直接把圖片傳給 Gemini ---
                # 這樣 Gemini 就不會被 YOLO 的標籤誤導，會自己看圖
                prompt = "你是一位專業的科學家。請觀察這張圖片，精確辨識其中的實驗物體、食物或器材，並給出：1. 該物體的科學名稱 2. 一個有趣的科學冷知識 3. 未來的創新應用建議。請用繁體中文回答。"
                
                response = gemini_model.generate_content([prompt, input_img])
                
                st.success("推論完成！")
                st.markdown(response.text)
