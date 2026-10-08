"""Local CLIP loading: never silently switch pretrained assets."""
def load_tokenizer(path):
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(path, local_files_only=True)


def write_cv2_image(path, pixels):
    """Preserve OpenCV encoding while supporting Windows Unicode paths."""
    from pathlib import Path
    import cv2
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, pixels)
    if not success:
        raise OSError(f'Image encoding failed: {path}')
    encoded.tofile(path)
