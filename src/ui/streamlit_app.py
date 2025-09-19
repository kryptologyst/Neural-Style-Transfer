"""
Modern Streamlit UI for Neural Style Transfer application.
"""

import streamlit as st
import requests
import io
from PIL import Image
import time
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.neural_style_transfer import StyleTransferPipeline
from database.models import DatabaseManager

# Page config
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .result-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .method-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transfer_history' not in st.session_state:
    st.session_state.transfer_history = []

if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'style_pipeline' not in st.session_state:
    st.session_state.style_pipeline = StyleTransferPipeline()

def main():
    st.markdown('<h1 class="main-header">🎨 Neural Style Transfer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose a page", [
            "Style Transfer", 
            "Template Gallery", 
            "History", 
            "Statistics",
            "About"
        ])
    
    if page == "Style Transfer":
        style_transfer_page()
    elif page == "Template Gallery":
        template_gallery_page()
    elif page == "History":
        history_page()
    elif page == "Statistics":
        statistics_page()
    elif page == "About":
        about_page()

def style_transfer_page():
    st.header("Create Your Stylized Image")
    
    # Method selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("🚀 Fast Transfer (AdaIN)")
        st.write("Quick results in seconds using Adaptive Instance Normalization")
        use_adain = st.button("Use Fast Method", key="adain_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("🎯 High Quality (Optimization)")
        st.write("Better quality results using iterative optimization")
        use_optimization = st.button("Use High Quality", key="opt_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if use_adain or use_optimization:
        method = "adain" if use_adain else "optimization"
        transfer_interface(method)

def transfer_interface(method):
    st.subheader(f"Style Transfer - {method.upper()}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📷 Content Image")
        content_file = st.file_uploader(
            "Upload your content image",
            type=['png', 'jpg', 'jpeg'],
            key="content_upload"
        )
        if content_file:
            content_image = Image.open(content_file)
            st.image(content_image, caption="Content Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("🎨 Style Image")
        
        # Option to use template or upload
        style_option = st.radio("Choose style source:", ["Upload Image", "Use Template"])
        
        if style_option == "Upload Image":
            style_file = st.file_uploader(
                "Upload your style image",
                type=['png', 'jpg', 'jpeg'],
                key="style_upload"
            )
            if style_file:
                style_image = Image.open(style_file)
                st.image(style_image, caption="Style Image", use_column_width=True)
        else:
            templates = st.session_state.db_manager.get_style_templates()
            if templates:
                template_names = [f"{t['name']} ({t['category']})" for t in templates]
                selected_template = st.selectbox("Choose a template:", template_names)
                template_idx = template_names.index(selected_template)
                template = templates[template_idx]
                
                # Display template (if image exists)
                st.write(f"**{template['name']}**")
                st.write(template['description'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Parameters
    st.subheader("⚙️ Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if method == "adain":
            alpha = st.slider("Style Strength (α)", 0.0, 1.0, 1.0, 0.1)
        else:
            steps = st.slider("Optimization Steps", 100, 1000, 300, 50)
    
    with col2:
        image_size = st.selectbox("Output Size", [256, 512, 1024], index=1)
    
    with col3:
        preserve_colors = st.checkbox("Preserve Original Colors")
    
    # Transfer button
    if st.button("🎨 Generate Stylized Image", type="primary"):
        if content_file and (style_file if style_option == "Upload Image" else selected_template):
            with st.spinner("Creating your stylized image..."):
                try:
                    # Save uploaded files temporarily
                    content_path = f"temp_content_{int(time.time())}.jpg"
                    content_image.save(content_path)
                    
                    if style_option == "Upload Image":
                        style_path = f"temp_style_{int(time.time())}.jpg"
                        style_image.save(style_path)
                    else:
                        style_path = template['style_image_path']
                    
                    output_path = f"output_{int(time.time())}.jpg"
                    
                    # Perform style transfer
                    start_time = time.time()
                    
                    if method == "adain":
                        result_path = st.session_state.style_pipeline.transfer_style_adain(
                            content_path, style_path, output_path, alpha
                        )
                    else:
                        result_path = st.session_state.style_pipeline.transfer_style_optimization(
                            content_path, style_path, output_path, steps
                        )
                    
                    processing_time = time.time() - start_time
                    
                    # Display result
                    st.success(f"✅ Style transfer completed in {processing_time:.2f} seconds!")
                    
                    result_image = Image.open(result_path)
                    st.image(result_image, caption="Stylized Result", use_column_width=True)
                    
                    # Save to database
                    parameters = {
                        "alpha": alpha if method == "adain" else None,
                        "steps": steps if method == "optimization" else None,
                        "image_size": image_size,
                        "preserve_colors": preserve_colors
                    }
                    
                    result_id = st.session_state.db_manager.save_transfer_result(
                        content_path, style_path, result_path, method, parameters, processing_time
                    )
                    
                    # Download button
                    with open(result_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Result",
                            data=file.read(),
                            file_name=f"stylized_image_{result_id}.jpg",
                            mime="image/jpeg"
                        )
                    
                    # Cleanup temporary files
                    for temp_file in [content_path, output_path]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    if style_option == "Upload Image" and os.path.exists(style_path):
                        os.remove(style_path)
                
                except Exception as e:
                    st.error(f"❌ Error during style transfer: {str(e)}")
        else:
            st.warning("⚠️ Please upload both content and style images!")

def template_gallery_page():
    st.header("🖼️ Style Template Gallery")
    
    templates = st.session_state.db_manager.get_style_templates()
    
    if not templates:
        st.info("No templates available. Add some templates to get started!")
        return
    
    # Category filter
    categories = list(set(t['category'] for t in templates if t['category']))
    selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    
    if selected_category != "All":
        templates = [t for t in templates if t['category'] == selected_category]
    
    # Display templates in grid
    cols = st.columns(3)
    
    for i, template in enumerate(templates):
        with cols[i % 3]:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            
            # Display template image if exists
            if os.path.exists(template['style_image_path']):
                template_image = Image.open(template['style_image_path'])
                st.image(template_image, caption=template['name'], use_column_width=True)
            else:
                st.write("🖼️ Image not found")
            
            st.write(f"**{template['name']}**")
            st.write(f"*{template['category']}*")
            st.write(template['description'])
            st.write(f"Popularity: {template['popularity_score']:.1f}")
            
            if st.button(f"Use Template", key=f"template_{template['id']}"):
                st.session_state.selected_template = template
                st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def history_page():
    st.header("📚 Transfer History")
    
    results = st.session_state.db_manager.get_transfer_results(limit=50)
    
    if not results:
        st.info("No transfer history available.")
        return
    
    # Display results
    for result in results:
        with st.expander(f"Transfer #{result['id']} - {result['method'].upper()} - {result['created_at'][:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Content Image**")
                if os.path.exists(result['content_image_path']):
                    content_img = Image.open(result['content_image_path'])
                    st.image(content_img, use_column_width=True)
            
            with col2:
                st.write("**Style Image**")
                if os.path.exists(result['style_image_path']):
                    style_img = Image.open(result['style_image_path'])
                    st.image(style_img, use_column_width=True)
            
            with col3:
                st.write("**Result**")
                if os.path.exists(result['output_image_path']):
                    result_img = Image.open(result['output_image_path'])
                    st.image(result_img, use_column_width=True)
                    
                    # Download button
                    with open(result['output_image_path'], "rb") as file:
                        st.download_button(
                            label="📥 Download",
                            data=file.read(),
                            file_name=f"result_{result['id']}.jpg",
                            mime="image/jpeg",
                            key=f"download_{result['id']}"
                        )
            
            # Metadata
            st.write(f"**Method:** {result['method']}")
            st.write(f"**Processing Time:** {result['processing_time']:.2f}s")
            st.write(f"**Parameters:** {result['parameters']}")

def statistics_page():
    st.header("📊 Application Statistics")
    
    results = st.session_state.db_manager.get_transfer_results(limit=1000)
    templates = st.session_state.db_manager.get_style_templates()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transfers", len(results))
    
    with col2:
        st.metric("Available Templates", len(templates))
    
    with col3:
        if results:
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        else:
            st.metric("Avg Processing Time", "N/A")
    
    with col4:
        methods_used = {}
        for result in results:
            method = result['method']
            methods_used[method] = methods_used.get(method, 0) + 1
        
        if methods_used:
            most_popular = max(methods_used, key=methods_used.get)
            st.metric("Most Popular Method", most_popular.upper())
        else:
            st.metric("Most Popular Method", "N/A")
    
    # Charts
    if results:
        st.subheader("Method Usage")
        st.bar_chart(methods_used)
        
        st.subheader("Processing Times")
        processing_times = [r['processing_time'] for r in results[-20:]]  # Last 20
        st.line_chart(processing_times)

def about_page():
    st.header("ℹ️ About Neural Style Transfer")
    
    st.markdown("""
    ## What is Neural Style Transfer?
    
    Neural Style Transfer is a technique that uses deep neural networks to combine the content of one image 
    with the artistic style of another image. This creates a new image that maintains the structure and 
    objects of the content image while adopting the artistic characteristics of the style image.
    
    ## Methods Available
    
    ### 🚀 Fast Transfer (AdaIN)
    - **Speed:** Very fast (seconds)
    - **Quality:** Good
    - **Technique:** Adaptive Instance Normalization
    - **Best for:** Quick previews, real-time applications
    
    ### 🎯 High Quality (Optimization)
    - **Speed:** Slower (minutes)
    - **Quality:** Excellent
    - **Technique:** Iterative optimization with VGG19
    - **Best for:** Final high-quality results
    
    ## Features
    
    - 🎨 Multiple style transfer algorithms
    - 📚 Pre-built style templates
    - 💾 Transfer history and statistics
    - 📱 Modern responsive UI
    - 🔄 Batch processing capabilities
    - 📊 Performance analytics
    
    ## Technical Details
    
    This application uses:
    - **PyTorch** for deep learning operations
    - **VGG19** pre-trained network for feature extraction
    - **AdaIN** for fast style transfer
    - **Streamlit** for the user interface
    - **SQLite** for data persistence
    
    ## Tips for Best Results
    
    1. **Content Images:** Use high-resolution images with clear subjects
    2. **Style Images:** Choose images with distinctive artistic styles
    3. **Parameters:** Experiment with different alpha values for style strength
    4. **Size:** Larger images take longer but produce better quality
    
    ---
    
    **Version:** 2.0.0  
    **Last Updated:** 2024
    """)

if __name__ == "__main__":
    main()
