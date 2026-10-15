"""Image file I/O adapter; preprocessing remains in the source pipelines."""
def write_cv2_image(path, pixels):
    """Preserve OpenCV encoding while supporting Windows Unicode paths."""
    from pathlib import Path
    import cv2
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, pixels)
    if not success:
        raise OSError(f'Image encoding failed: {path}')
    encoded.tofile(path)
