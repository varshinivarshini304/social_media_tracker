import streamlit as st
import base64
import io
import json
import os
import random
import cv2
import numpy as np
import easyocr
from PIL import Image
from datetime import datetime
import re
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Social Mood Matcher Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Social Mood Matcher Pro\nThis is an enhanced AI-powered social media content generator."
    }
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
    * {
        box-sizing: border-box;
    }
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-family: 'Playfair Display', serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .caption-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        line-height: 1.6;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hashtag-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: #0d47a1;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        height: 3rem;
        width: 100%;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .stExpander {
        background: rgba(255,255,255,0.5);
        border-radius: 12px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .error-box {
        background: #fed7d7;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: #c6f6d5;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #feebc8;
        border-left: 4px solid #dd6b20;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
    }
    .stAlert {
        border-radius: 12px;
    }
    textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2d3748;
    }
    .stDivider {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.markdown('<h1 class="main-title">🎭 Social Mood Matcher Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Caption & Hashtag Generator with Advanced OCR & Emotion Analysis</p>', unsafe_allow_html=True)
st.divider()

# ====================== SESSION STATE ======================
if 'result' not in st.session_state:
    st.session_state.result = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# ====================== API KEY MANAGEMENT ======================
def get_api_key():
    """Securely retrieve API key from environment or secrets."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            return api_key
        if hasattr(st, "secrets") and "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
        return None
    except Exception:
        return None

api_key = get_api_key()

if 'api_valid' not in st.session_state:
    st.session_state.api_valid = True

if 'api_error_message' not in st.session_state:
    st.session_state.api_error_message = None

def mark_api_invalid(message):
    st.session_state.api_valid = False
    st.session_state.api_error_message = message

# ====================== OCR INITIALIZATION (FIXED) ======================
@st.cache_resource
def init_ocr_reader():
    """Initialize EasyOCR reader with caching to prevent re-loading."""
    try:
        return easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception as e:
        st.error(f"Failed to initialize OCR: {str(e)}")
        return None

# ====================== IMAGE PREPROCESSING (FIXED) ======================
def preprocess_image_for_ocr(image):
    """Advanced image preprocessing to improve OCR accuracy."""
    try:
        if isinstance(image, Image.Image):
            if image.mode not in ['L', 'RGB']:
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        enhanced = cv2.equalizeHist(binary)
        height, width = enhanced.shape
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return enhanced
    except Exception as e:
        st.warning(f"Image preprocessing failed: {str(e)}. Using original image.")
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                return np.array(image.convert('L'))
            return np.array(image)
        return image

# ====================== TEXT EXTRACTION (FIXED) ======================
def extract_text_from_image(uploaded_file):
    """Extract text from image using EasyOCR with preprocessing."""
    try:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image_for_ocr(image)
        reader = init_ocr_reader()
        if reader is None:
            return "Unable to initialize OCR. Please check the image and try again."
        results = reader.readtext(processed_image)
        extracted_texts = [text for (bbox, text, confidence) in results if confidence > 0.3]
        combined_text = ' '.join(extracted_texts)
        if not combined_text.strip():
            if image.mode != 'L':
                original_gray = np.array(image.convert('L'))
            else:
                original_gray = np.array(image)
            results = reader.readtext(original_gray)
            extracted_texts = [text for (bbox, text, confidence) in results if confidence > 0.3]
            combined_text = ' '.join(extracted_texts)
        return combined_text.strip() if combined_text.strip() else "No readable text found in the image."
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def correct_ocr_text(text):
    """Apply common OCR spell-fix corrections locally."""
    if not text or text.startswith("Error"):
        return text
    corrected = text.strip()
    replacements = {
        'YQUCANMACIEISREAL': 'YOU CAN IMAGINE IS REAL',
        'CANMACIE': 'CAN IMAGINE',
        'YOUCAN': 'YOU CAN',
        'YQU': 'YOU',
        'IMACINE': 'IMAGINE',
        'MACIE': 'IMAGINE',
        'EVIERYHN': 'EVERYTHING',
        'EVERYHN': 'EVERYTHING',
        'ISREAL': 'IS REAL',
        'TOU': 'YOU',
        'IY': 'MY',
        'EVA': 'EVERY'
    }
    for wrong, right in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        corrected = re.sub(re.escape(wrong), right, corrected, flags=re.IGNORECASE)
    corrected = re.sub(r'\s{2,}', ' ', corrected).strip()
    return corrected

# ====================== EMOTION ANALYSIS ======================
EMOTION_KEYWORDS = {
    'happy': ['happy', 'joy', 'joyful', 'love', 'lovely', 'excited', 'yay', 'smile', 'great', 'amazing', 
              'wonderful', 'fantastic', 'excellent', 'brilliant', 'awesome', 'beautiful', 'good', 'best',
              'glad', 'cheerful', 'delighted', 'pleased', 'ecstatic', 'elated', 'thrilled'],
    'sad': ['sad', 'sadness', 'sorrow', 'unhappy', 'tears', 'cry', 'depressed', 'miserable', 'heartbroken', 
            'grief', 'lonely', 'desperate', 'hopeless', 'blue', 'down', 'gloomy', 'mourn', 'regret'],
    'angry': ['angry', 'mad', 'furious', 'annoyed', 'rage', 'irritated', 'frustrated', 'livid', 'enraged', 
              'incensed', 'outraged', 'seething', 'bitter', 'hostile', 'resentful', 'exasperated'],
    'surprised': ['surprise', 'surprised', 'wow', 'shocked', 'astonished', 'amazed', 'stunned', 'flabbergasted', 
                  'dumbfounded', 'bewildered', 'unexpected', 'startled', 'awestruck'],
    'fear': ['fear', 'scared', 'afraid', 'nervous', 'anxious', 'terrified', 'panic', 'dread', 'horror', 'terror', 
             'petrified', 'horrified', 'worried', 'apprehensive', 'uneasy', 'frightened']
}

def analyze_emotion_lexicon(text):
    """Analyze emotion using keyword matching."""
    if not text or text == "No readable text found in the image.":
        return {'sentiment': 'Neutral', 'emotion': 'Neutral', 'scores': {}, 'confidence': 0.5}
    text_l = text.lower()
    scores = {k: 0 for k in EMOTION_KEYWORDS}
    for emo, kws in EMOTION_KEYWORDS.items():
        for kw in kws:
            scores[emo] += text_l.count(kw)
    best_emotion = max(scores, key=lambda k: scores[k])
    total_score = sum(scores.values())
    if total_score == 0:
        sentiment = 'Neutral'
        emotion = 'Neutral'
        confidence = 0.5
    else:
        emotion = best_emotion.capitalize()
        confidence = scores[best_emotion] / total_score if total_score > 0 else 0.5
        if emotion in ['Happy', 'Surprised']:
            sentiment = 'Positive'
        elif emotion in ['Sad', 'Angry', 'Fear']:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    return {
        'sentiment': sentiment, 
        'emotion': emotion, 
        'scores': scores,
        'confidence': confidence,
        'primary_emotion': best_emotion if total_score > 0 else 'neutral'
    }

def analyze_emotion_with_api(text, api_key):
    """Analyze emotion using DeepSeek API with fallback."""
    if not api_key or not text:
        return analyze_emotion_lexicon(text)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        prompt = f"""Analyze the following text and return ONLY a valid JSON object with exactly these keys:
- sentiment: (one of: \"Positive\", \"Neutral\", \"Negative\")
- emotion: (one word like Happy, Sad, Angry, Fear, Surprised, Neutral)
- confidence: (a number between 0 and 1)
- explanation: (brief 1-sentence explanation)
Text to analyze: \"{text[:500]}\"
Return ONLY the JSON object, no other text."""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'emotion': result.get('emotion', 'Neutral'),
                'confidence': result.get('confidence', 0.7),
                'explanation': result.get('explanation', '')
            }
        else:
            return analyze_emotion_lexicon(text)
    except Exception as e:
        mark_api_invalid(str(e))
        st.warning(f"API emotion analysis failed: {str(e)}. Using local analysis.")
        return analyze_emotion_lexicon(text)

# ====================== CAPTION GENERATION ======================
HUMAN_RESPONSE_TEMPLATES = {
    'casual': [
        "Sundays like this make me smile a little wider.",
        "Just here soaking up the little moments that feel like magic.",
        "Feeling kind of blissed out today.",
        "This one made me stop and appreciate how good life can be.",
        "Low-key loving this vibe right now.",
        "Caught a feeling I didn't know I needed.",
        "Simple joy, big heart.",
        "Good mood, good light, good company.",
        "Some days are just soft and sweet.",
        "Taking a minute to savor this one.",
        "This picture really sums up that cozy energy.",
        "Smiling at how nice this moment felt.",
        "Grateful for these quiet, perfect moments.",
        "Life's little treasures make everything worthwhile.",
        "Finding beauty in the ordinary, joy in the simple."
    ],
    'aesthetic': [
        "Soft textures and quiet colors making everything feel calm.",
        "A little slice of beauty in the everyday.",
        "Chasing that dreamlike light and gentle moods.",
        "There's something poetic about this scene.",
        "When the world feels curated just for you.",
        "Warm tones and slow moments are my favorite.",
        "This is the kind of view you want to hold on to.",
        "Aesthetic energy, muted colors, and real feels.",
        "This one feels like a memory in the making.",
        "Soft, simple, and full of meaning.",
        "A small moment looking beautifully quiet.",
        "Just here for the moody, dreamy feels.",
        "Art in the ordinary, beauty in the mundane.",
        "Capturing the ethereal in everyday moments.",
        "Where silence speaks louder than words."
    ],
    'professional': [
        "Grateful for the progress and momentum today.",
        "Celebrating the work that feels intentional and smart.",
        "Focused on growth and the wins that come with it.",
        "A clean moment of clarity in a busy week.",
        "Proud of staying disciplined and moving forward.",
        "Working hard, staying thoughtful, and celebrating small wins.",
        "This one feels like a step in the right direction.",
        "Creating space for hustle and meaningful impact.",
        "Leadership looks a lot like consistency and care.",
        "A polished moment that matches the energy behind it.",
        "Driven by ideas, not just deadlines.",
        "A snapshot of focus, purpose, and quiet ambition.",
        "Building something meaningful, one step at a time.",
        "Success is the sum of small efforts repeated daily.",
        "Professional excellence meets personal fulfillment."
    ],
    'playful': [
        "This one's giving instant sparkle and good energy.",
        "Can't stop smiling at how fun today felt.",
        "Just vibing with all the joyful chaos.",
        "Bringing the energy and a whole lot of laughs.",
        "This moment was 100% mood.",
        "Playful and bright, just how I like it.",
        "Life's better with a little bit of sparkle.",
        "Laughing my way through the good vibes.",
        "This one felt like a tiny celebration.",
        "Pure fun wrapped up in one photo.",
        "Here for the color, joy, and happy noise.",
        "Keeping it light, bright, and a little wild.",
        "Dancing through life with joy in my heart.",
        "Spending my days in a state of joyful rebellion.",
        "Unapologetically me, unreservedly happy."
    ]
}
EMOTION_TO_STYLE = {
    'happy': 'casual',
    'joyful': 'casual',
    'excited': 'playful',
    'surprised': 'playful',
    'sad': 'aesthetic',
    'fear': 'aesthetic',
    'angry': 'professional',
    'neutral': 'casual'
}

def generate_caption(emotion_data, style=None):
    """Generate a caption based on emotion analysis."""
    emotion = emotion_data.get('emotion', 'Neutral').lower()
    sentiment = emotion_data.get('sentiment', 'Neutral')
    if not style:
        style = EMOTION_TO_STYLE.get(emotion, 'casual')
    templates = HUMAN_RESPONSE_TEMPLATES.get(style, HUMAN_RESPONSE_TEMPLATES['casual'])
    base_caption = random.choice(templates)
    if sentiment == 'Positive':
        enhancers = [" ✨", " 🌟", " 💫", " 🎉", " 💕"]
        base_caption += random.choice(enhancers)
    elif sentiment == 'Negative':
        enhancers = [" 💭", " 🕊️", " 🌙", " 💫", " 🌧️"]
        base_caption += random.choice(enhancers)
    return base_caption

def generate_caption_with_api(text, emotion_data, api_key):
    """Generate caption using DeepSeek API."""
    if not api_key:
        return generate_caption(emotion_data)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        prompt = f"""Generate a single, engaging social media caption based on this text and emotion:
Text: \"{text[:300]}\"
Detected emotion: {emotion_data.get('emotion', 'Neutral')}
Sentiment: {emotion_data.get('sentiment', 'Neutral')}
Requirements:
- Keep it under 200 characters
- Be authentic and conversational
- Don't use hashtags in the caption
- Return ONLY the caption text
Caption:"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        caption = response.choices[0].message.content.strip()
        return caption if caption else generate_caption(emotion_data)
    except Exception as e:
        mark_api_invalid(str(e))
        st.warning(f"API caption generation failed: {str(e)}. Using template.")
        return generate_caption(emotion_data)

# ====================== HASHTAG GENERATION ======================
HASHTAG_CATEGORIES = {
    'casual': ["#GoodVibes", "#Beautiful", "#HappyMoments", "#ChillDay", "#EverydayJoy", 
               "#LifeIsGood", "#PositiveVibes", "#Grateful", "#SimplePleasures", "#DailyJoy",
               "#Blessed", "#WellnessJourney", "#MindfulLiving", "#SlowLiving", "#CozyVibes"],
    'aesthetic': ["#Aesthetic", "#Dreamy", "#MoodyGrams", "#SoftVibes", "#VisualPoetry", 
                  "#ArtisticSoul", "#CreativeVision", "#BeautyInDetails", "#MindfulLiving", 
                  "#SoulfulArt", "#AestheticEdit", "#ArtOfVisuals", "#MoodInColors", 
                  "#VibeWithMe", "#ArtisticExpression"],
    'professional': ["#Success", "#Motivation", "#Growth", "#ProfessionalLife", "#Leadership", 
                     "#CareerGoals", "#BusinessMindset", "#WorkEthic", "#ProfessionalDevelopment", 
                     "#SuccessMindset", "#HustleSmart", "#CareerGrowth", "#LeadershipQualities", 
                     "#ProfessionalExcellence", "#BusinessSuccess"],
    'playful': ["#FunVibes", "#Lit", "#GoodTimes", "#PlayfulMood", "#Joyful", "#LaughOutLoud", 
                "#FunTimes", "#HappyPlace", "#JoyfulLiving", "#PlayfulSpirit", "#LivingMyBestLife", 
                "#NoBadDays", "#SmileMore", "#GoodEnergy", "#PureJoy"]
}
EMOTION_HASHTAGS = {
    'happy': ["#Joyful", "#HappyHeart", "#SmileEveryday", "#PositiveEnergy", "#Blissful"],
    'sad': ["#DeepThoughts", "#Reflective", "#QuietMoments", "#SoulSearching", "#Emotional"],
    'angry': ["#StandingUp", "#SpeakingTruth", "#RealTalk", "#Unfiltered", "#Authentic"],
    'surprised': ["#Shook", "#Unexpected", "#MindBlown", "#WowMoment", "#PlotTwist"],
    'fear': ["#Courage", "#FacingFears", "#BraveHeart", "#InnerStrength", "#GrowthMindset"]
}

def generate_hashtags(emotion_data, style, num_hashtags=10):
    """Generate relevant hashtags based on emotion and style."""
    emotion = emotion_data.get('primary_emotion', 'neutral')
    style_tags = HASHTAG_CATEGORIES.get(style, HASHTAG_CATEGORIES['casual'])
    emotion_tags = EMOTION_HASHTAGS.get(emotion, [])
    all_tags = list(dict.fromkeys(style_tags + emotion_tags))
    random.shuffle(all_tags)
    return all_tags[:num_hashtags]

# ====================== DEMO MODE ======================
def get_demo_result(style, num_hashtags):
    """Generate demo content when API is unavailable."""
    emotion_data = {
        'sentiment': 'Positive' if style in ['casual', 'playful'] else 'Calm' if style == 'aesthetic' else 'Professional',
        'emotion': 'Happy' if style in ['casual', 'playful'] else 'Calm' if style == 'aesthetic' else 'Focused',
        'primary_emotion': 'happy' if style in ['casual', 'playful'] else 'neutral' if style == 'aesthetic' else 'angry'
    }
    caption = generate_caption(emotion_data, style)
    hashtags = HASHTAG_CATEGORIES.get(style, HASHTAG_CATEGORIES['casual'])[:num_hashtags]
    return {
        'sentiment': emotion_data['sentiment'],
        'emotion': emotion_data['emotion'],
        'caption': caption,
        'hashtags': hashtags,
        'full_post': f"{caption}\n\n{' '.join(hashtags)}"
    }

# ====================== MAIN PROCESSING FUNCTION ======================
def process_image(uploaded_file, use_api, num_hashtags, style_override, api_key):
    """Main processing pipeline."""
    status_placeholder = st.empty()
    try:
        status_placeholder.info("📸 Extracting text from image...")
        extracted_text = extract_text_from_image(uploaded_file)
        corrected_text = correct_ocr_text(extracted_text)
        final_text = corrected_text or extracted_text
        if corrected_text != extracted_text:
            status_placeholder.info("🛠️ Correcting OCR output...")
            time.sleep(0.3)
        time.sleep(0.5)
        status_placeholder.info("🎭 Analyzing emotional tone...")
        if use_api and api_key and st.session_state.api_valid:
            emotion_data = analyze_emotion_with_api(final_text, api_key)
        else:
            emotion_data = analyze_emotion_lexicon(final_text)
        time.sleep(0.5)
        status_placeholder.info("✍️ Crafting the perfect caption...")
        if use_api and api_key and st.session_state.api_valid:
            caption = generate_caption_with_api(final_text, emotion_data, api_key)
        else:
            caption = generate_caption(emotion_data, style_override)
        time.sleep(0.5)
        status_placeholder.info("#️⃣ Generating relevant hashtags...")
        hashtags = generate_hashtags(emotion_data, style_override, num_hashtags)
        time.sleep(0.5)
        result = {
            'raw_text': extracted_text,
            'corrected_text': final_text,
            'sentiment': emotion_data['sentiment'],
            'emotion': emotion_data['emotion'],
            'confidence': emotion_data.get('confidence', 0.7),
            'caption': caption,
            'hashtags': hashtags,
            'full_post': f"{caption}\n\n{' '.join(hashtags)}"
        }
        status_placeholder.success("✅ Processing complete!")
        return result
    except Exception as e:
        status_placeholder.error(f"❌ Processing failed: {str(e)}")
        return None

# ====================== MAIN UI ======================
def main():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if not api_key:
            st.warning("⚠️ No API key found. Please set DEEPSEEK_API_KEY in environment variables or .streamlit/secrets.toml")
            st.info("💡 Get a free API key from DeepSeek to unlock AI features")
        elif not st.session_state.api_valid:
            st.error("⚠️ DeepSeek API key is invalid or expired. Using local fallback.")
            st.info("Update DEEPSEEK_API_KEY and reload to restore API features.")
        use_api = st.checkbox("🤖 Use AI API", value=bool(api_key) and st.session_state.api_valid, help="Enable for AI-powered content generation (requires API key)")
        if use_api and not api_key:
            use_api = False
        style_override = st.selectbox(
            "🎨 Content Style",
            ["casual", "aesthetic", "professional", "playful"],
            index=0,
            help="Choose the tone of your content"
        )
        num_hashtags = st.slider("#️⃣ Number of hashtags", 5, 20, 10)
        st.divider()
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Upload an image** containing text
            2. **AI extracts** text using OCR
            3. **Emotion analysis** detects mood
            4. **Smart generation** creates captions & hashtags
            5. **Copy & share** your perfect post!
            💡 **Pro tips:**
            - Clear, well-lit images work best
            - Handwriting is supported
            - API provides better results
            - Try different styles for variety
            """)
        st.divider()
        if st.session_state.result:
            st.markdown("### 📊 Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", st.session_state.result['sentiment'])
            with col2:
                st.metric("Emotion", st.session_state.result['emotion'])
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image with text",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload clear images for best OCR results"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.markdown("### 🎯 Generate Content")
        if st.button("✨ Generate Caption & Hashtags", type="primary", disabled=not uploaded_file):
            if uploaded_file:
                with st.spinner("Processing your image..."):
                    result = process_image(uploaded_file, use_api, num_hashtags, style_override, api_key)
                    if result:
                        st.session_state.result = result
                        st.session_state.error_message = None
                    else:
                        st.session_state.error_message = "Processing failed. Please try again."
        if st.session_state.result:
            result = st.session_state.result
            st.success(f"🎉 Content generated! Detected: {result['emotion']} mood")
            st.markdown("#### 📝 Caption")
            st.markdown(f'<div class="caption-card">{result["caption"]}</div>', unsafe_allow_html=True)
            st.markdown("#### #️⃣ Hashtags")
            hashtags_html = " ".join([f'<span style="display: inline-block; background: #e3f2fd; padding: 5px 12px; border-radius: 20px; margin: 5px; color: #0d47a1;">{tag}</span>' for tag in result['hashtags']])
            st.markdown(f'<div class="hashtag-card">{hashtags_html}</div>', unsafe_allow_html=True)
            st.markdown("#### 📱 Preview Full Post")
            st.code(result['full_post'], language="text")
            st.info("💡 Click the copy icon in the top-right corner of the code block above to copy the text.")
            if result.get('raw_text') and result['raw_text'] != "No readable text found in the image.":
                with st.expander("🔍 View OCR Results"):
                    st.info(f"**OCR Result:** {result['raw_text']}")
                    if result.get('corrected_text') and result['corrected_text'] != result['raw_text']:
                        st.success(f"**Corrected Text:** {result['corrected_text']}")
            if result.get('confidence'):
                st.progress(result['confidence'], text=f"Confidence: {result['confidence']:.0%}")
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #718096; font-size: 0.8rem;">
            🎭 Social Mood Matcher Pro | Powered by AI & Computer Vision
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
