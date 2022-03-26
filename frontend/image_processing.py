import io
from PIL import Image
from PIL.ExifTags import TAGS


def correct_orientation(image: Image) -> Image:
    orientation = None
    for key, value in image.getexif().items():
        if TAGS.get(key) == "Orientation":
            orientation = value

    if orientation == 3:
        image = image.transpose(Image.ROTATE_180)
    if orientation == 6:
        image = image.transpose(Image.ROTATE_270)
    if orientation == 8:
        image = image.transpose(Image.ROTATE_90)

    return image


def compress_image(image: Image) -> bytes:
    width, height = image.size
    image = image.resize((int(width*.4), int(height*.4)),
                         resample=Image.ANTIALIAS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=40)
    image_bytes = buffer.getvalue()
    return image_bytes


def process_image(image: bytes) -> bytes:
    """Correct image orientation, and compress it if the image 
        is larger than 5MB."""
    image_byte_size = len(image)
    image = Image.open(io.BytesIO(image))

    image = correct_orientation(image)

    if image_byte_size >= 5042880:
        return compress_image(image)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    return image_bytes
