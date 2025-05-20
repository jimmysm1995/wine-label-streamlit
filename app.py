import streamlit as st
import asyncio
from ultralytics import YOLO
from PIL import Image
import io
from bson import ObjectId
from pymongo import MongoClient
import env  # asegúrate de que el archivo exista o usa dotenv
from azure_vision_ocr import extract_text_from_image
from faiss_search import WineSearchResult
from preprocess_image import preprocess_image_for_ocr
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = MongoClient(env.MONGO_CONECTION_STRING)
db = client["RequestsDb"]
collection = db['wines']
model = YOLO('runs/detect/wine_detector/weights/best.pt')

async def main():
    st.title("Sube una imagen")

    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_container_width=True)  # usa use_container_width

        if st.button("Buscar vino en la imagen"):
            try:
                processed_bytes = preprocess_image_for_ocr(uploaded_file.getvalue())
                image_for_detection = Image.open(io.BytesIO(uploaded_file.getvalue()))

                results = model(image_for_detection)

                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()

                conf_threshold = 0.6
                detecciones_validas = [conf for conf in confidences if conf > conf_threshold]

                image_with_boxes = results[
                    0].plot()  # Devuelve una imagen con las detecciones dibujadas (como array numpy)
                image_with_boxes_pil = Image.fromarray(image_with_boxes)
                st.image(image_with_boxes_pil, caption="Detecciones de vino", use_container_width=True)

                if len(detecciones_validas) == 0:
                    st.warning("La imagen no parece ser un vino.")
                    st.stop()
                else:
                    processed_stream = io.BytesIO(processed_bytes)
                    wineScanResult = await extract_text_from_image(processed_stream)

                    if isinstance(wineScanResult, WineSearchResult):
                        wine_id = wineScanResult.matched_ids[0]
                        wine_data = collection.find_one({"_id": ObjectId(wine_id)})

                        if wine_data:
                            st.write(f"**Nombre:** {wine_data.get('name')}")
                            st.write(f"**Bodega:** {wine_data.get('winery')}")
                            st.write(f"**Precio:** {wine_data.get('price')} €")
                        else:
                            st.warning("No se encontró el vino en la base de datos.")
                    else:
                        st.error("No se reconoció ningún vino.")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

# Ejecutar el async main correctamente
if __name__ == "__main__":
    asyncio.run(main())
