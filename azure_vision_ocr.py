import asyncio
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

import env
from faiss_search import search_wine_by_name

azure_key = env.AZURE_VISION_KEY
azure_endpoint = env.AZURE_VISION_ENDPOINT
# Configurar el cliente de Azure
credentials = CognitiveServicesCredentials(azure_key)
computervision_client = ComputerVisionClient(azure_endpoint, credentials)

async def extract_text_from_image(image_bytes: BytesIO):
    read_response = computervision_client.read_in_stream(image_bytes, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    # Esperar a que la operación de lectura se complete
    while True:
        read_result = computervision_client.get_read_result(operation_id)  # Removed 'await'
        if read_result.status not in ['notStarted', 'running']:
            break
        await asyncio.sleep(1)

    # Extraer los textos y calcular su tamaño
    extracted_texts = []
    if read_result.status == "succeeded":
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                # Calcular el tamaño del texto basado en bounding_box
                bbox = line.bounding_box
                width = abs(bbox[2] - bbox[0])  # Diferencia en X
                height = abs(bbox[5] - bbox[1])  # Diferencia en Y
                area = width * height  # Aproximación del tamaño del texto
                extracted_texts.append((line.text, area))

    # Ordenar por tamaño en orden descendente
    extracted_texts.sort(key=lambda x: x[1], reverse=True)

    # Extraer solo los textos (sin las áreas)
    lines = [text for text, _ in extracted_texts]
    full_text = " ".join(lines[:3])

    print("Texto extraído:", full_text)

    # Buscar IDs del vino si el nombre es válido
    return await search_wine_by_name(full_text)