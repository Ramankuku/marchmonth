import streamlit as st
import requests
import base64
import pandas as pd
import uuid

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agentic AI Data Scientist", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

def render_result(r):
    try:
        if not isinstance(r, dict):
            st.write(str(r))
            return

        r_type = r.get("type", "text")

        if r_type == "plot" and "b64" in r:
            try:
                img = base64.b64decode(r["b64"])
                st.image(img, use_container_width=True)
            except:
                st.warning("⚠️ Failed to render plot")

        elif r_type == "table" and "data" in r:
            try:
                df = pd.DataFrame(r["data"])
                st.dataframe(df, use_container_width=True)
            except:
                st.warning("⚠️ Invalid table data")

        elif r_type == "text":
            st.write(r.get("content", "⚠️ Empty response"))

        else:
            st.write("⚠️ Unknown result:", r)

    except Exception as e:
        st.error(f"Rendering error: {e}")


with st.sidebar:
    st.title("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            uploaded_file.seek(0)

            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

            if df.empty or df.shape[1] == 0:
                st.error("❌ Empty or invalid CSV")
            else:
                st.session_state.df_preview = df
                st.success("✅ File loaded")

                uploaded_file.seek(0)

                with st.spinner("Uploading..."):
                    res = requests.post(
                        f"{API}/upload",
                        files={"file": uploaded_file}
                    )

                data = res.json()

                if data.get("status") == "success":
                    st.session_state.file_path = data["file_path"]
                    st.success("✅ Uploaded successfully")
                else:
                    st.error(data.get("message", "Upload failed"))

        except pd.errors.EmptyDataError:
            st.error("❌ File is empty")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.session_state.df_preview is not None:
        st.subheader("📊 Preview")
        st.dataframe(st.session_state.df_preview.head(), use_container_width=True)


st.title("🚀 Agentic AI Data Scientist")
st.caption("Chat with your dataset")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        for r in msg.get("results", []):
            render_result(r)


query = st.chat_input("Ask something about your dataset...")

if query:

    if not st.session_state.file_path:
        st.warning("⚠️ Upload dataset first")
        st.stop()
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing..."):

            try:
                res = requests.post(
                    f"{API}/analyze",
                    json={
                        "session_id": st.session_state.session_id,
                        "file_path": st.session_state.file_path,
                        "query": query
                    }
                )

                data = res.json()

                if data.get("status") != "success":
                    st.error(data.get("message", "Error occurred"))
                else:
                    st.write(data.get("insights", "No insights"))

                    for r in data.get("results", []):
                        render_result(r)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data.get("insights", ""),
                        "results": data.get("results", [])
                    })

            except Exception as e:
                st.error(f"❌ API Error: {e}")