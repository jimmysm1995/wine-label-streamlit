from typing import List, TypeVar, Generic
import numpy as np

T = TypeVar("T")


class CandidatoSelector(Generic[T]):
    def __init__(self, max_z: float = 1.0, max_std: float = 0.03):
        """
        :param max_z: umbral máximo de z-score para considerar un candidato dentro del grupo de confianza
        :param max_std: si la desviación estándar total es menor que este valor, se devuelven todos los candidatos
        """
        self.max_z = max_z
        self.max_std = max_std

    def seleccionar(self, candidatos: List[T], distancias: List[float]) -> List[T]:
        if len(candidatos) != len(distancias) or not distancias:
            raise ValueError("Listas de candidatos y distancias deben coincidir y no estar vacías.")

        distancias = np.array(distancias)
        media = np.mean(distancias)
        std = np.std(distancias) or 1e-6  # evita división por cero

        # Si todos los candidatos están muy cerca entre sí, devuélvelos todos
        if std < self.max_std:
            return candidatos

        # Calcula z-score relativo para cada distancia
        z_scores = (distancias - media) / std

        # Devuelve los candidatos dentro del rango de confianza
        seleccionados = [
            candidato for candidato, z in zip(candidatos, z_scores) if z <= self.max_z
        ]

        return seleccionados