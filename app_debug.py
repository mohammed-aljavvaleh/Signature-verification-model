
import streamlit as st

st.title("🔍 Debug Test")

try:

    st.write("Step 1: Importing sys...")

    import sys

    sys.path.insert(0, '.')

    st.success("✅ sys imported")

    

    st.write("Step 2: Importing Config...")

    from src.config import Config

    st.success("✅ Config imported")

    

    st.write("Step 3: Importing Model...")

    from src.model import SiameseNetwork

    st.success("✅ Model imported")

    

    st.write("Step 4: Checking model file...")

    model_path = Config.MODELS_DIR / Config.BEST_MODEL_NAME

    st.write(f"Looking for: {model_path}")

    st.write(f"Exists: {model_path.exists()}")

    

    if model_path.exists():

        st.success("✅ Model file found!")

    else:

        st.error("❌ Model file not found!")

    

except Exception as e:

    st.error(f"❌ Error: {str(e)}")

    import traceback

    st.code(traceback.format_exc())

st.write("✅ Debug complete!")

