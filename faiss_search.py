import asyncio
import os
from datetime import datetime
from typing import List, Dict
import faiss
import numpy as np
import openai
from openai import AzureOpenAI
from pymongo import MongoClient
import pickle
import re
import unicodedata
from azure.storage.blob import BlobServiceClient
import env
from bson import ObjectId
from dataclasses import dataclass
import asyncio
from condidato_selector import CandidatoSelector
from fuzzywuzzy import fuzz

@dataclass
class FilteredMatchResult():
    id: ObjectId
    score: float
    name: str

@dataclass
class MatchResult():
    id: ObjectId
    distance: float

@dataclass
class WineSearchResult():
    matched_ids: List[str]
    wine_name: str

def file_needs_update(local_path, blob_client):
    """Compara tamaños para verificar si el archivo ha cambiado."""
    if not os.path.exists(local_path):
        return True  # No existe localmente, necesita descargarse

    local_size = os.path.getsize(local_path)
    blob_size = blob_client.get_blob_properties().size
    return local_size != blob_size  # Si los tamaños son distintos, hay actualización

client = MongoClient(env.MONGO_CONECTION_STRING)
db = client["RequestsDb"]
collection_wine = db['wines']
blob_string = env.BLOB_STORAGE_STRING
container_name = env.CONTAINER_FAISS_NAME
blob_faiss_name = env.BLOB_FAISS_NAME

# Crear cliente de Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(blob_string)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_faiss_name)

# Descargar solo si ha cambiado
if file_needs_update(blob_faiss_name, blob_client):
    print("Descargando índice FAISS actualizado desde Azure...")
    with open(blob_faiss_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

# Cargar el índice FAISS
faiss_index = faiss.read_index(blob_faiss_name)

# Cargar los IDs de MongoDB (suponiendo que los guardaste en un archivo)
with open("wine_ids.pkl", "rb") as f:
    wine_ids = pickle.load(f)  # Lista de IDs de MongoDB


STOPWORDS_ES = {"el", "la", "los", "las", "y", "de", "en", "con", "para",
                "a", "por", "que", "un", "una", "al", "del", "como", "se", "su",
                "más", "si", "no", "bodegas", "vinos", "vino", "bodega",
                "cosecha", "añada", "cl", "ml", "vol", "375", "75", "50", "x4", "x3", "x2", "1l"
                ,"seleccion", "denominacion", "origen", "cabernet", "sauvignon", "merlot",}

async def generate_embedding(text: str) -> np.ndarray:
    """Genera un embedding basado en el nombre del vino y la cosecha."""

    if not text.strip():
        raise ValueError("El texto para generar embedding está vacío.")

    # Validación de API Key
    api_key = env.AZURE_EMBEDDING_KEY
    base_url = env.AZURE_EMBEDDING_URL
    api_version = env.AZURE_EMBEDDING_VERSION

    if not all([api_key, base_url, api_version]):
        raise ValueError("Faltan credenciales de OpenAI en las variables de entorno.")

    client = AzureOpenAI(api_key=api_key, api_version=api_version, base_url=base_url)

    try:
        response = await asyncio.to_thread(client.embeddings.create, model="text-embedding-3-small", input=text)
        return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
    except Exception as e:
        print(f"Error al generar embedding: {e}")
        return np.zeros((1, 384), dtype="float32")

UMBRAL = 0.5  # 50% de similitud global mínima
ABSOLUTE_THRESHOLD = 0.7 # 80% de similitud global mínima


def calcular_pesos_dinamicos(ocr_texto: str)-> Dict[str, float]:
        """Calcula pesos dinámicos para los campos en base a la longitud del texto OCR."""
        palabras = clean_wine_name(ocr_texto).split()
        longitud = len(palabras)

        # Inicializar pesos base
        pesos = {
            "name": 0.7,
            "winery": 0.1,
            "denomination": 0.1,
            "type": 0.1,
        }

        # Ajustar según la cantidad de palabras (cuanto más corto, más confiamos en 'name')
        if longitud <= 2:
            pesos["name"] = 0.85
            pesos["winery"] = 0.05
            pesos["denomination"] = 0.05
            pesos["type"] = 0.05
        elif longitud <= 4:
            pesos["name"] = 0.75
            pesos["winery"] = 0.1
            pesos["denomination"] = 0.1
            pesos["type"] = 0.05
        elif longitud <= 7:
            pesos["name"] = 0.6
            pesos["winery"] = 0.15
            pesos["denomination"] = 0.15
            pesos["type"] = 0.1
        else:
            pesos["name"] = 0.5
            pesos["winery"] = 0.2
            pesos["denomination"] = 0.2
            pesos["type"] = 0.1

        return pesos


def calcular_similitud(ocr_texto: str, vino: dict) -> float:
    ocr_texto = clean_wine_name(ocr_texto)
    total_peso = 0
    puntuacion = 0
    
    pesos = calcular_pesos_dinamicos(ocr_texto)

    for campo, peso in pesos.items():
        valor = vino.get(campo, "")
        valor_norm = clean_wine_name(valor)
        similitud = (
                            fuzz.partial_ratio(valor_norm, ocr_texto) +
                            fuzz.token_sort_ratio(valor_norm, ocr_texto) +
                            fuzz.token_set_ratio(valor_norm, ocr_texto)
                    ) / 3 / 100
        puntuacion += similitud * peso
        total_peso += peso

    return puntuacion / total_peso if total_peso > 0 else 0

def calcular_umbral_dinamico(ocr_texto: str) -> float:
    longitud = len(ocr_texto.strip().split())

    if longitud <= 2:
        return 0.7
    elif longitud <= 5:
        return 0.5
    else:
        return 0.4

def clean_wine_name(name: str) -> str:
    """Limpia y normaliza el nombre del vino."""
    if not name or not isinstance(name, str):
        return ""

    name = name.lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')  # Elimina acentos
    name = re.sub(r"[^a-z0-9 ]", "", name)  # Elimina caracteres especiales
    words = name.split()

    # Usar un conjunto para eliminar duplicados mientras mantenemos el orden
    seen = set()
    cleaned_words = []
    for word in words:
        if word not in seen and word not in STOPWORDS_ES:
            cleaned_words.append(word)
            seen.add(word)

    # Unir las palabras en una cadena sin modificar el orden original
    clean_name = " ".join(cleaned_words)
    return clean_name.strip()


def calcular_umbral_faiss_dinamico(wine_name: str) -> float:
    palabras = len(wine_name.split())
    return max(0.75, 0.9 - palabras * 0.015)

def process_text(query: str):
    clean_query = clean_wine_name(query)
    return clean_query

async def search_faiss(wine_name: str):
    query_vector = await generate_embedding(wine_name)
    distances, indices = faiss_index.search(query_vector, k=3)

    if indices[0][0] == -1 or distances[0][0] > calcular_umbral_faiss_dinamico(wine_name):
        return [], []

    matches = [
        MatchResult(id=wine_ids[idx], distance=dist)
        for idx, dist in zip(indices[0], distances[0])
        if idx != -1 and ObjectId.is_valid(wine_ids[idx])
    ]
    first_match = matches.copy()
    return matches, first_match

def filter_by_stats(matches: list[MatchResult]) -> list[MatchResult]:
    dist_list = [m.distance for m in matches]
    selector = CandidatoSelector(max_z=1.0, max_std=0.03)
    return selector.seleccionar(matches, dist_list)

async def search_wine_by_name(
        query: str,
        threshold: float = 0.2
) -> WineSearchResult:

    try:
        # 1. Limpiar y extraer info del texto
        clean_query = process_text(query)

        # 2. Embedding + búsqueda FAISS
        matches, first_match = await search_faiss(clean_query)

        print(matches)
        # 3. Filtrado estadístico
        selected_matches = filter_by_stats(matches)

        matched_ids_only = [w.id for w in selected_matches]

        return WineSearchResult(
            matched_ids=[str(_id) for _id in matched_ids_only],
            wine_name=clean_query
        )
    except Exception as e:
        print(e)
        return WineSearchResult(matched_ids=[],wine_name="")

def find_wine_mongo(
        wine_ids: List[str],
) -> List[str]:
    results = []
    for _id in wine_ids:
        wine = collection_wine.find_one({"_id": ObjectId(_id)})
        results.append(wine["name"] + ", " + wine["winery"])
        if not wine:
            continue
    return results