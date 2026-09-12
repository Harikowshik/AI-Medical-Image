import os
import io
import tempfile
from PIL import Image as PILImage, ImageEnhance
import streamlit as st

# Agno and Google Gemini imports
from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# Page configuration
st.set_page_config(
    page_title="Medical Imaging Diagnosis Agent",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical UI styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header card styling */
    .medical-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .medical-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.25);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 10px;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .medical-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .medical-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }

    /* Info card */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        color: #e2e8f0;
    }

    /* Disclaimer box */
    .disclaimer-card {
        background-color: rgba(234, 88, 12, 0.1);
        border-left: 4px solid #f97316;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 0.88rem;
        color: #fed7aa;
    }
    
    /* Result card */
    .report-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if "GOOGLE_API_KEY" not in st.session_state or not st.session_state.GOOGLE_API_KEY:
    st.session_state.GOOGLE_API_KEY = DEFAULT_API_KEY

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")
    
    # API Key Handling
    api_key_input = st.text_input(
        "Google AI Studio API Key:",
        type="password",
        value=st.session_state.GOOGLE_API_KEY if st.session_state.GOOGLE_API_KEY else "",
        placeholder="Enter your AI Studio API key...",
        help="Required for accessing Google Gemini vision models."
    )
    
    if api_key_input != st.session_state.GOOGLE_API_KEY:
        st.session_state.GOOGLE_API_KEY = api_key_input
        st.rerun()

    if not st.session_state.GOOGLE_API_KEY:
        st.caption(
            "🔑 Don't have a key? Get one for free at "
            "[Google AI Studio](https://aistudio.google.com/apikey) (up to 1,500 free requests/day)"
        )
    else:
        st.success("✅ Google Gemini API Key Active")

    st.markdown("---")
    
    # Gemini Model Selection
    st.markdown("### 🤖 Vision Model")
    model_options = {
        "gemini-3.6-flash": "Gemini 3.6 Flash (Recommended: State-of-the-Art Vision)",
        "gemini-2.5-flash": "Gemini 2.5 Flash (Ultra-Fast Multimodal)",
        "gemini-3.7-flash": "Gemini 3.7 Flash (Next-Gen Multimodal)",
        "gemini-2.5-pro": "Gemini 2.5 Pro (Deep Clinical Reasoning)"
    }
    selected_model_id = st.selectbox(
        "Select Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
        help="Gemini 3.6 Flash offers the latest features, superior speed, and accurate diagnostic analysis for medical scans."
    )

    # Capabilities & Grounding
    st.markdown("### 🛠️ Agent Capabilities")
    enable_grounding = st.checkbox(
        "Enable Google Search Grounding",
        value=False,
        help="Enables Google Search grounding for real-time medical literature. Leave off for fastest, direct offline-capable multimodal vision."
    )
    
    analysis_focus = st.selectbox(
        "Diagnostic Protocol Focus:",
        options=[
            "Standard Comprehensive Radiology",
            "Emergency / Acute Triage",
            "Oncological / Lesion Screening",
            "Musculoskeletal & Orthopedic",
            "Patient-Centered Clarity"
        ],
        index=0
    )

    st.markdown("---")
    
    # Clinical Disclaimer
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Clinical Disclaimer:</strong><br>
        This application is designed for educational, research, and assistive exploration only. 
        All AI-generated interpretations must be reviewed by board-certified radiologists or licensed healthcare professionals. 
        Do not use as a sole diagnostic basis for medical treatments.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.session_state.GOOGLE_API_KEY and st.button("🔄 Reset API Key", use_container_width=True):
        st.session_state.GOOGLE_API_KEY = ""
        st.session_state.analysis_result = None
        st.rerun()

# Main Header Banner
st.markdown("""
<div class="medical-header">
    <div class="medical-badge">Autonomous Diagnostic Radiologist Agent</div>
    <div class="medical-title">🩻 Medical Imaging Diagnosis Agent</div>
    <div class="medical-subtitle">
        Powered by <strong>Agno Agent Framework</strong> and <strong>Google Gemini 2.0 Flash</strong> multimodal reasoning.
        Inspect X-rays, MRI scans, CT slices, and Ultrasound with clinical rigor.
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to load and normalize images (including DICOM)
def process_uploaded_image(file_or_path, is_sample=False):
    dicom_meta = None
    if not is_sample and hasattr(file_or_path, "name") and file_or_path.name.lower().endswith((".dcm", ".dicom")):
        if not HAS_PYDICOM:
            st.error("pydicom is required to parse DICOM files.")
            return None, None
        try:
            dcm = pydicom.dcmread(file_or_path)
            arr = dcm.pixel_array.astype(float)
            p_min, p_max = arr.min(), arr.max()
            if p_max > p_min:
                normalized = ((arr - p_min) / (p_max - p_min) * 255.0).astype("uint8")
            else:
                normalized = arr.astype("uint8")
            image = PILImage.fromarray(normalized).convert("RGB")
            dicom_meta = {
                "Modality": getattr(dcm, "Modality", "Unknown"),
                "Patient ID": getattr(dcm, "PatientID", "De-identified"),
                "Study Date": getattr(dcm, "StudyDate", "N/A"),
                "Body Part": getattr(dcm, "BodyPartExamined", "N/A"),
                "Photometric Interpretation": getattr(dcm, "PhotometricInterpretation", "N/A"),
            }
            return image, dicom_meta
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            return None, None
    else:
        try:
            image = PILImage.open(file_or_path).convert("RGB")
            return image, None
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return None, None

# Initialize Agent
def create_medical_agent(api_key, model_id, search_enabled=False):
    if not api_key:
        return None
    try:
        return Agent(
            model=Gemini(
                id=model_id,
                api_key=api_key,
                search=search_enabled
            ),
            markdown=True
        )
    except Exception:
        # Fallback without search if model/backend doesn't support search param
        return Agent(
            model=Gemini(
                id=model_id,
                api_key=api_key
            ),
            markdown=True
        )

medical_agent = create_medical_agent(
    st.session_state.GOOGLE_API_KEY,
    selected_model_id,
    enable_grounding
)

# Application Layout
col_left, col_right = st.columns([1.1, 1.4], gap="large")

with col_left:
    st.markdown("### 📤 1. Medical Scan Input")
    
    # Input selection: Upload file OR use sample
    input_source = st.radio(
        "Choose Scan Source:",
        ["📁 Upload Image / DICOM", "🧪 Use Preloaded Clinical Sample"],
        horizontal=True
    )
    
    raw_image = None
    dicom_meta = None
    image_identifier = None
    
    if input_source == "📁 Upload Image / DICOM":
        uploaded_file = st.file_uploader(
            "Upload Scan (X-Ray, CT, MRI, Ultrasound)",
            type=["jpg", "jpeg", "png", "dicom", "dcm"],
            help="Supported formats: JPG, JPEG, PNG, DICOM (.dcm)"
        )
        if uploaded_file is not None:
            raw_image, dicom_meta = process_uploaded_image(uploaded_file, is_sample=False)
            image_identifier = uploaded_file.name
    else:
        sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        sample_choice = st.selectbox(
            "Select a Clinical Sample Scan:",
            [
                "Chest X-Ray (PA View - Thoracic Evaluation)",
                "Brain MRI (Axial T2-Weighted Neuroimaging)"
            ]
        )
        
        sample_map = {
            "Chest X-Ray (PA View - Thoracic Evaluation)": os.path.join(sample_dir, "sample_chest_xray.jpg"),
            "Brain MRI (Axial T2-Weighted Neuroimaging)": os.path.join(sample_dir, "sample_brain_mri.jpg")
        }
        sample_path = sample_map[sample_choice]
        if os.path.exists(sample_path):
            raw_image, dicom_meta = process_uploaded_image(sample_path, is_sample=True)
            image_identifier = sample_choice
        else:
            st.warning("Sample image not found on disk.")

    # Image Display & Radiological Adjustments
    if raw_image is not None:
        st.markdown("#### 🔬 Radiologist Viewport")
        
        # Display DICOM metadata if available
        if dicom_meta:
            with st.expander("📋 DICOM Metadata Header", expanded=False):
                for k, v in dicom_meta.items():
                    st.write(f"**{k}:** `{v}`")

        # Viewport adjustments
        with st.expander("🎛️ Contrast & Windowing Adjustments", expanded=False):
            contrast_val = st.slider("Contrast", 0.5, 2.5, 1.0, 0.1)
            brightness_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            sharpness_val = st.slider("Edge Sharpness", 0.5, 2.5, 1.0, 0.1)

        # Apply adjustments
        adjusted_image = raw_image.copy()
        if contrast_val != 1.0:
            adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(contrast_val)
        if brightness_val != 1.0:
            adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(brightness_val)
        if sharpness_val != 1.0:
            adjusted_image = ImageEnhance.Sharpness(adjusted_image).enhance(sharpness_val)

        # Resize for consistent display and inference
        w, h = adjusted_image.size
        display_width = 540
        display_height = int(display_width * (h / w))
        resized_for_display = adjusted_image.resize((display_width, display_height))

        st.image(
            resized_for_display,
            caption=f"Selected Scan: {image_identifier}",
            use_container_width=True
        )

        # Trigger Analysis Button
        analyze_clicked = st.button(
            "🩺 Perform Clinical Diagnostic Analysis",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.GOOGLE_API_KEY)
        )
        
        if not st.session_state.GOOGLE_API_KEY:
            st.warning("⚠️ Please provide a Google AI Studio API key in the sidebar to enable analysis.")
    else:
        st.info("👆 Please upload a medical scan or select a preloaded sample to begin.")
        analyze_clicked = False

with col_right:
    st.markdown("### 📋 2. Diagnostic Assessment & Findings")

    # Construct the clinical diagnostic query
    clinical_query = f"""
You are a board-certified radiologist and clinical imaging specialist with dual expertise in diagnostic radiology and evidence-based clinical medicine. 
Analyze the provided medical scan carefully following the "{analysis_focus}" protocol.

Structure your radiological assessment comprehensively using the following markdown format:

### 1. Imaging Modality & Technical Evaluation
- **Modality**: Identify the scan type (e.g., Radiograph/X-ray, CT slice, MRI sequence like T1/T2/FLAIR, Ultrasound, etc.).
- **Anatomical Region & Projection**: Specify anatomical boundaries, projection/plane (e.g., PA upright, AP supine, axial, sagittal, coronal).
- **Technical Adequacy**: Quality of exposure, penetration, patient positioning, inspiratory effort, artifacts, or limitations.

### 2. Systematic Key Findings
- **Primary Observations**: Systematically inspect organ systems / structural compartments (e.g., in chest: lungs/pleura, mediastinum/cardiomediastinal contour, osseous structures, soft tissues).
- **Abnormality Localization**: Precise anatomical site, margins (well-defined vs ill-defined), morphology, density/attenuation/signal intensity.
- **Measurements / Dimensions**: Specific estimated dimensions or standard radiological indices where applicable.
- **Severity Rating**: Classify as [Normal / Mild / Moderate / Severe / Critical Emergency].

### 3. Diagnostic Impression & Differential Diagnosis
- **Primary Diagnosis**: Most likely diagnosis with clinical reasoning and estimated diagnostic confidence level (%).
- **Differential Diagnoses**: List 2-3 alternative differential diagnoses in order of likelihood, detailing what features support or detract from each.
- **Urgent / Critical Alerts**: Clearly highlight if any acute life-threatening signs are detected (e.g., tension pneumothorax, midline shift, acute intracranial hemorrhage, bowel perforation). If absent, state "No critical red-flag findings identified".

### 4. Patient-Friendly Summary
- Provide a clear, empathetic explanation written in 6th-to-8th grade reading level for the patient and their family.
- Avoid or translate complex clinical jargon into accessible visual analogies.
- Address typical questions: "What does this mean for me?" and "What are the recommended next steps to discuss with my doctor?".

### 5. Research Context & Clinical Protocols
- Summarize 2-3 standard evidence-based medical guidelines and treatment protocols (e.g., ACR appropriateness criteria, WHO, or clinical consensus).
- Cite standard medical literature benchmarks or typical diagnostic workup steps for similar clinical presentations.

Be clinically precise, objective, and maintain highest medical documentation standards.
"""

    if analyze_clicked and raw_image is not None and medical_agent:
        with st.spinner(f"🔍 Analyzing scan with {selected_model_id}... Evaluating radiological criteria..."):
            temp_file = None
            try:
                # Resize optimal for Gemini multimodal token efficiency
                max_dim = 1024
                scale = min(max_dim / raw_image.width, max_dim / raw_image.height, 1.0)
                optimized_size = (int(raw_image.width * scale), int(raw_image.height * scale))
                optimized_img = raw_image.resize(optimized_size)

                # Resilient Dual-Engine Execution:
                output_text = ""
                temp_path = None
                
                # 1. Attempt via Agno Agent Framework (with disk tempfile if space permits)
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        temp_path = tf.name
                    optimized_img.save(temp_path, format="PNG")
                    agno_img = AgnoImage(filepath=temp_path)

                    if medical_agent:
                        run_output = medical_agent.run(
                            clinical_query,
                            images=[agno_img]
                        )
                        output_text = str(run_output.content) if run_output and run_output.content else ""
                except Exception as agno_err:
                    output_text = ""

                # 2. In-Memory Direct Inference Fallback:
                # If Agno was skipped, encountered [Errno 28] (no space), network/SSL drop, or 404
                is_err = (
                    not output_text 
                    or len(output_text.strip()) < 40 
                    or any(kw in output_text for kw in ["[Errno", "getaddrinfo", "[SSL:", "EOF occurred", "TimeoutException", "404", "NOT_FOUND", "not found"])
                )

                if is_err:
                    # Direct in-memory Google GenAI Client (requires 0 bytes of disk space!)
                    from google import genai
                    client = genai.Client(
                        api_key=st.session_state.GOOGLE_API_KEY,
                        http_options={'api_version': 'v1beta'}
                    )
                    gen_response = client.models.generate_content(
                        model=selected_model_id,
                        contents=[clinical_query, optimized_img]
                    )
                    if gen_response and gen_response.text:
                        output_text = gen_response.text

                st.session_state.analysis_result = output_text
                st.session_state.last_image_name = image_identifier

            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("Tip: Verify your API key has quota enabled at https://aistudio.google.com")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    # Display results if available
    if st.session_state.analysis_result:
        st.success(f"✅ Diagnostic Analysis Complete for `{st.session_state.last_image_name}` (Model: `{selected_model_id}`)")
        
        tab_report, tab_patient, tab_export = st.tabs([
            "📑 Full Radiological Report",
            "🩺 Patient-Friendly Guidance",
            "💾 Export & Download"
        ])
        
        with tab_report:
            st.markdown(st.session_state.analysis_result)
        
        with tab_patient:
            st.info("💡 Below is an excerpt highlighting key patient-focused guidance from the diagnostic assessment:")
            # Extract or display patient friendly section
            content = st.session_state.analysis_result
            if "### 4. Patient-Friendly" in content:
                parts = content.split("### 4. Patient-Friendly")
                patient_part = parts[1].split("### 5.")[0] if len(parts) > 1 else parts[0]
                st.markdown("### 🩺 Patient-Friendly Summary" + patient_part)
            else:
                st.markdown(content)
        
        with tab_export:
            st.markdown("#### 📥 Download Clinical Documentation")
            st.write("Export the generated diagnostic report for patient records or referral review:")
            
            report_text = f"""# Medical Imaging Diagnosis Report
**Scan Reference:** {st.session_state.last_image_name}
**Analysis Model:** {selected_model_id}
**Focus Protocol:** {analysis_focus}

---
{st.session_state.analysis_result}

---
*Disclaimer: Generated by AI-assisted diagnostic agent. For informational and educational purposes only. Requires verification by a licensed healthcare provider.*
"""
            safe_img_id = (image_identifier or st.session_state.last_image_name or "scan").replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            st.download_button(
                label="📄 Download Full Markdown Report (.md)",
                data=report_text,
                file_name=f"medical_report_{safe_img_id}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
            st.download_button(
                label="📝 Download Plain Text Report (.txt)",
                data=report_text,
                file_name=f"medical_report_{safe_img_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class="report-card">
            <h4 style="color: #94a3b8; margin-top: 0;">Awaiting Scan Analysis</h4>
            <p style="color: #64748b; font-size: 0.95rem;">
                Select or upload a medical image on the left, verify your Google AI Studio API key in the sidebar, 
                and click <strong>"Perform Clinical Diagnostic Analysis"</strong> to generate an in-depth radiological evaluation.
            </p>
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px dashed #334155; border-radius: 8px; padding: 20px; text-align: center; color: #94a3b8;">
                <span style="font-size: 2.2rem;">🩺</span>
                <p style="margin: 8px 0 0 0; font-size: 0.9rem;">
                    Supports Chest X-rays, Brain MRIs, Abdominal CTs, Orthopedic scans, & DICOM files.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
