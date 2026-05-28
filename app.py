import streamlit as st
import base64
import io
import json
import os
from PIL import Image
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Social Mood Matcher",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }
    .caption-card {
        background: #f8f9fa;
        padding: 1.8rem;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .hashtag-card {
        background: #e3f2fd;
        padding: 1.3rem;
        border-radius: 16px;
        color: #0d47a1;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        height: 3rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.markdown('<h1 class="main-title">🎭 Social Mood Matcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Caption & Hashtag Generator using DeepSeek Vision</p>', unsafe_allow_html=True)
st.divider()

# ====================== SESSION STATE ======================
if 'result' not in st.session_state:
    st.session_state.result = None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    style = st.selectbox(
        "🎨 Caption Style",
        ["casual", "aesthetic", "professional", "playful"]
    )
    
    platform = st.selectbox(
        "📱 Platform",
        ["Instagram", "Twitter/X", "Facebook", "LinkedIn"]
    )
    
    num_hashtags = st.slider(
        "#️⃣ Number of Hashtags",
        min_value=4, max_value=12, value=6
    )
    
    st.divider()
    
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    
    if api_key:
        st.success("✅ DeepSeek API Connected")
    else:
        st.warning("⚠️ API Key Not Found")
        st.info("""
        **How to add API Key:**
        1. Go to Settings → Secrets
        2. Add key name: `DEEPSEEK_API_KEY`
        3. Paste your key from platform.deepseek.com
        """)

# ====================== MAIN AREA ======================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("🚀 Generate with DeepSeek", type="primary"):
            if not api_key:
                st.error("Please add your DeepSeek API key in Secrets")
            else:
                with st.spinner("🔍 DeepSeek Vision is analyzing your image..."):
                    try:
                        from openai import OpenAI
                        
                        # Convert image to base64
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='JPEG', quality=85)
                        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                        
                        client = OpenAI(
                            api_key=api_key,
                            base_url="https://api.deepseek.com/v1"
                        )
                        
                        prompt = f"""Analyze this image and return **ONLY** valid JSON object:
{{
    "sentiment": "Happy | Calm | Aesthetic | Energetic | Cozy | Nostalgic",
    "category": "scenery | food | people | pet | urban | nature | abstract | travel",
    "caption": "Write a {style} style caption in 1-2 engaging sentences suitable for {platform}",
    "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6"]
}}"""

                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                                ]
                            }],
                            temperature=0.75,
                            max_tokens=450
                        )
                        
                        text = response.choices[0].message.content.strip()
                        
                        # Clean JSON response
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0]
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0]
                        
                        result = json.loads(text)
                        
                        hashtags = result.get('hashtags', [])[:num_hashtags]
                        caption = result.get('caption', 'Beautiful moment captured ✨')
                        full_post = f"{caption}\n\n{' '.join(hashtags)}"
                        
                        st.session_state.result = {
                            'sentiment': result.get('sentiment'),
                            'category': result.get('category'),
                            'caption': caption,
                            'hashtags': hashtags,
                            'full_post': full_post
                        }
                        st.success("✅ Successfully Generated!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
                        st.info("Using demo mode...")
                        st.session_state.result = get_demo_result(style, num_hashtags)
                        st.rerun()

with col2:
    st.subheader("✨ Generated Content")
    
    if st.session_state.result:
        r = st.session_state.result
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🎭 Sentiment", r.get('sentiment', '—'))
        with col_b:
            st.metric("📂 Category", r.get('category', '—'))
        
        st.markdown(f'<div class="caption-card">{r["caption"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hashtag-card">{" ".join(r["hashtags"])}</div>', unsafe_allow_html=True)
        
        st.text_area("📋 Copy Full Post:", r['full_post'], height=150)
        
        st.download_button(
            label="💾 Download as TXT",
            data=r['full_post'],
            file_name=f"social_post_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("👆 Upload an image and click **Generate** to start")

st.divider()
st.caption("Powered by DeepSeek Vision AI • Social Mood Matcher")

# ====================== DEMO FUNCTION ======================
def get_demo_result(style, num_hashtags):
    demos = {
        "casual": {
            "sentiment": "Happy",
            "category": "Lifestyle",
            "caption": "Loving this beautiful vibe! ✨ Such a perfect moment captured 🥰",
            "hashtags": ["#GoodVibes", "#Beautiful", "#HappyMoments", "#LifeIsGood", "#PositiveVibes"]
        },
        "aesthetic": {
            "sentiment": "Calm",
            "category": "Art",
            "caption": "Soft tones, gentle light, and endless serenity. Living in this dreamy moment 🌸",
            "hashtags": ["#Aesthetic", "#Dreamy", "#MoodyGrams", "#VisualPoetry", "#CalmVibes"]
        },
        "professional": {
            "sentiment": "Professional",
            "category": "Business",
            "caption": "A moment of excellence captured perfectly. Quality speaks louder than words.",
            "hashtags": ["#Professional", "#SuccessMindset", "#BusinessLife", "#Growth", "#Inspiration"]
        },
        "playful": {
            "sentiment": "Energetic",
            "category": "Fun",
            "caption": "OMG this is literally everything! 🔥 Can't stop smiling 😍",
            "hashtags": ["#FunVibes", "#HappyEnergy", "#PartyMood", "#Lit", "#GoodTimes"]
        }
    }
    
    result = demos.get(style, demos["casual"])
    result['hashtags'] = result['hashtags'][:num_hashtags]
    result['full_post'] = f"{result['caption']}\n\n{' '.join(result['hashtags'])}"
    return result
