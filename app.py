import asyncio
from ultralytics import YOLO
from PIL import Image
import io
from bson import ObjectId
from pymongo import MongoClient
import env
from azure_vision_ocr import extract_text_from_image
from faiss_search import WineSearchResult
from preprocess_image import preprocess_image_for_ocr
import os
import streamlit as st
import numpy as np

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
        st.image(image, caption="Imagen subida", use_container_width=True)

        if st.button("Buscar vino en la imagen"):
            try:
                image_for_detection = Image.open(io.BytesIO(uploaded_file.getvalue()))
                results = model(image_for_detection)
                boxes = results[0].boxes

                if not boxes:
                    st.warning("No se detectaron etiquetas de vino.")
                    st.stop()

                # Buscar el cuadro más grande (área)
                max_area = 0
                biggest_box = None
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        biggest_box = (int(x1), int(y1), int(x2), int(y2))

                if biggest_box is None:
                    st.warning("No se detectó ninguna botella de vino con confianza suficiente.")
                    st.stop()

                # Recortar la región del cuadro más grande
                cropped_image = image_for_detection.crop(biggest_box)

                # Procesar imagen para OCR
                buf = io.BytesIO()
                cropped_image.save(buf, format='JPEG')
                processed_bytes = preprocess_image_for_ocr(buf.getvalue())
                processed_stream = io.BytesIO(processed_bytes)

                wineScanResult = await extract_text_from_image(processed_stream)

                if isinstance(wineScanResult, WineSearchResult) and wineScanResult.matched_ids:
                    st.write(f"Texto extraido de la imagen: {wineScanResult.wine_name}")
                    st.success(f"Se encontraron {len(wineScanResult.matched_ids)} coincidencias:")

                    for wine_id in wineScanResult.matched_ids:
                        wine_data = collection.find_one({"_id": ObjectId(wine_id)})
                        if wine_data:
                            st.markdown("---")
                            st.write(f"**Nombre:** {wine_data.get('name')}")
                            st.write(f"**Bodega:** {wine_data.get('winery')}")
                            st.write(f"**Precio:** {wine_data.get('price')} €")
                            st.write(f"**Tipo:** {wine_data.get('type')}")
                            st.write(f"**Maridaje:** {wine_data.get('pairing')}")
                            st.image(wine_data.get('image'), caption="Imagen del vino", use_container_width=True)
                        else:
                            st.warning(f"ID no encontrado: {wine_id}")
                else:
                    st.error("No se reconoció ningún vino.")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

if __name__ == "__main__":
    asyncio.run(main())

