import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="T5 Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS (FIXED - Added color property)
# =============================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        border: none;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
        color: #1a1a1a;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-header">📝 T5-Small Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fine-tuned with QLoRA on CNN/DailyMail Dataset</div>', unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the QLoRA fine-tuned T5-small model"""
    try:
        adapter_path = "./Model_Files"
        
        if not os.path.exists(adapter_path):
            return None, None, f"Model folder not found at '{adapter_path}'"
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-small",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        return model, tokenizer, None
        
    except Exception as e:
        return None, None, str(e)

# Load model
with st.spinner("🔄 Loading model... Please wait."):
    model, tokenizer, error = load_model()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("""
    **Please check:**
    1. The 'Model_Files' folder is in the same directory as app.py
    2. All model files are present in Model_Files folder
    """)
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# =============================================================================
# SIDEBAR - GENERATION SETTINGS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Generation Settings")
    st.markdown("---")
    
    max_length = st.slider(
        "📏 Max Summary Length",
        min_value=50,
        max_value=200,
        value=128,
        step=10,
        help="Maximum length of generated summary in tokens"
    )
    
    min_length = st.slider(
        "📐 Min Summary Length",
        min_value=20,
        max_value=100,
        value=40,
        step=10,
        help="Minimum length of generated summary in tokens"
    )
    
    num_beams = st.slider(
        "🔍 Number of Beams",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Higher values = better quality but slower"
    )
    
    length_penalty = st.slider(
        "⚖️ Length Penalty",
        min_value=0.5,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Controls summary length preference"
    )
    
    no_repeat_ngram = st.slider(
        "🚫 No Repeat N-gram Size",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Prevents repetition of word sequences"
    )
    
    st.markdown("---")
    st.markdown("""
    **📊 Model Information:**
    - **Base Model:** T5-Small (60M params)
    - **Method:** QLoRA Fine-tuning
    - **Dataset:** CNN/DailyMail
    - **Adapter Size:** ~3.4 MB
    - **Runtime:** CPU
    """)

# =============================================================================
# INITIALIZE SESSION STATE FOR EXAMPLES (FIXED)
# =============================================================================
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""

# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.markdown("---")

# Example texts dictionary
examples = {
    "Technology News": """Artificial intelligence continues to revolutionize industries worldwide with unprecedented speed. Recent developments in large language models have demonstrated remarkable capabilities in natural language understanding, code generation, and creative tasks. Major technology companies are investing billions of dollars in AI research and development, racing to create more powerful and efficient systems. These AI systems are being deployed across healthcare, education, finance, and transportation sectors. However, experts emphasize the critical importance of responsible AI development, including addressing algorithmic bias, ensuring transparency, and maintaining robust ethical guidelines. Governments and regulatory bodies are working to establish comprehensive frameworks for AI governance. The technology promises to enhance human capabilities while raising important questions about privacy, employment, and societal impact.""",
    
    "Health & Fitness": """Regular physical exercise provides numerous health benefits for people of all ages and fitness levels. Scientific studies consistently demonstrate that just 30 minutes of moderate physical activity daily can significantly reduce the risk of cardiovascular disease, type 2 diabetes, and certain types of cancer. Exercise strengthens the cardiovascular system, improves bone density, and enhances muscular strength and flexibility. Beyond physical benefits, regular activity has profound effects on mental health, reducing symptoms of depression and anxiety while improving mood and cognitive function. It also helps maintain healthy body weight, improves sleep quality, and boosts immune system function. Health professionals recommend combining aerobic activities like walking or cycling with strength training exercises and flexibility work for optimal health outcomes. Starting with small, manageable goals and gradually increasing intensity can help establish sustainable exercise habits.""",
    
    "Sports News": """The championship game concluded last night with a thrilling victory for the home team in front of a sold-out stadium. After trailing by 15 points at halftime, the team mounted an impressive comeback in the third quarter, displaying remarkable resilience and determination. The star player delivered an outstanding performance, scoring 42 points and grabbing 18 rebounds while providing crucial leadership during critical moments. The atmosphere was electric as fans witnessed one of the most memorable games in franchise history. This victory marks the team's first championship title in over two decades, ending a long drought that had tested the patience of devoted supporters. The coach praised the team's mental toughness and their ability to execute under pressure. Players celebrated emotionally with teammates, coaches, and fans who had supported them throughout the challenging season.""",
    
    "Business Article": """The global economy is experiencing significant transformation driven by technological innovation and changing consumer behaviors. Companies are rapidly adapting their business models to meet evolving market demands and remain competitive in an increasingly digital landscape. E-commerce platforms continue to gain market share as consumers embrace online shopping for convenience and variety. Traditional retailers are investing heavily in omnichannel strategies, integrating physical and digital experiences to serve customers effectively. Supply chain disruptions have prompted businesses to diversify their supplier networks and invest in resilient logistics systems. Sustainability has become a central concern, with companies implementing environmentally friendly practices to meet consumer expectations and regulatory requirements. Financial markets remain volatile as investors navigate uncertainty around interest rates, inflation, and geopolitical tensions. Business leaders emphasize the importance of agility and innovation in responding to rapid market changes."""
}

# Example selector
with st.expander("💡 Try an example text"):
    example_choice = st.selectbox(
        "Select an example:",
        ["-- Choose an example --", "Technology News", "Health & Fitness", "Sports News", "Business Article"]
    )
    
    if example_choice != "-- Choose an example --":
        if st.button("📋 Use this example", key="use_example"):
            st.session_state.example_text = examples[example_choice]
            st.rerun()

# Input text area (FIXED - uses session state)
text = st.text_area(
    "📄 Enter text to summarize:",
    value=st.session_state.example_text,  # Use session state value
    height=250,
    placeholder="Paste your article or text here...\n\nThe model works best with news articles, blog posts, and informative content.\n\nMinimum 10 words recommended.",
    help="Enter or paste the text you want to summarize",
    key="text_input"
)

# Clear example after use
if text != st.session_state.example_text and st.session_state.example_text != "":
    st.session_state.example_text = ""

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🚀 Generate Summary", use_container_width=True, type="primary")

# =============================================================================
# SUMMARY GENERATION
# =============================================================================
if generate_button:
    if not text or len(text.strip()) == 0:
        st.warning("⚠️ Please enter some text to summarize.")
    elif len(text.split()) < 10:
        st.warning("⚠️ Text is too short. Please enter at least 10 words for better results.")
    else:
        with st.spinner("🤖 Generating summary... This may take 10-30 seconds."):
            try:
                # Prepare input
                input_text = f"summarize: {text}"
                
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                
                # Generate summary
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram,
                        early_stopping=True,
                    )
                
                # Decode summary
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📄 Generated Summary")
                
                # Check if summary is empty
                if summary and len(summary.strip()) > 0:
                    # Summary in styled box (FIXED - now has color)
                    st.markdown(
                        f'<div class="summary-box">{summary}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error("⚠️ Model generated an empty summary. Please try again with different text.")
                
                # Statistics
                st.markdown("### 📊 Statistics")
                col1, col2, col3 = st.columns(3)
                
                input_words = len(text.split())
                summary_words = len(summary.split())
                compression_ratio = round((1 - summary_words / input_words) * 100, 1) if input_words > 0 else 0
                
                with col1:
                    st.metric(
                        label="📝 Input Length",
                        value=f"{input_words} words",
                        help="Number of words in the original text"
                    )
                
                with col2:
                    st.metric(
                        label="📋 Summary Length",
                        value=f"{summary_words} words",
                        help="Number of words in the generated summary"
                    )
                
                with col3:
                    st.metric(
                        label="📉 Compression",
                        value=f"{compression_ratio}%",
                        help="Percentage of text reduction"
                    )
                
                # Download button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")
                st.info("Try reducing the text length or adjusting the generation parameters in the sidebar.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <small>
            <b>T5-Small Summarizer</b> | Fine-tuned with QLoRA<br>
            Trained on CNN/DailyMail Dataset | Powered by Transformers & PEFT<br>
            <i>⚡ Running on CPU for universal compatibility</i>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)

# Debug information
if st.sidebar.checkbox("🔧 Show Debug Info", value=False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 Debug Information:**")
    st.sidebar.write(f"• Model device: {next(model.parameters()).device}")
    st.sidebar.write(f"• PyTorch version: {torch.__version__}")
    st.sidebar.write(f"• CUDA available: {torch.cuda.is_available()}")
    st.sidebar.write(f"• Model dtype: {next(model.parameters()).dtype}")
    if hasattr(model, 'peft_config'):
        st.sidebar.write("• ✅ LoRA adapters loaded")
    else:
        st.sidebar.write("• ❌ No LoRA adapters detected")
    st.sidebar.write(f"• Adapter path: ./Model_Files")
