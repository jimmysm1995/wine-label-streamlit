from PIL import Image, ImageEnhance, ImageOps
import io

def preprocess_image_for_ocr(image_bytes: bytes, output_format='JPEG', resize_max=1600) -> bytes:
    # Abrir imagen desde bytes
    image = Image.open(io.BytesIO(image_bytes))

    # 1. Rotación automática según EXIF
    image = ImageOps.exif_transpose(image)

    # 2. Redimensionar si es necesario
    if max(image.size) > resize_max:
        image.thumbnail((resize_max, resize_max), Image.LANCZOS)

    # 3. Convertir a escala de grises
    image = image.convert("L")

    # 4. Aumentar brillo/contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)

    # 5. Guardar en buffer
    output_buffer = io.BytesIO()
    image.save(output_buffer, format=output_format, quality=85 if output_format == 'JPEG' else None)
    output_buffer.seek(0)

    return output_buffer.read()