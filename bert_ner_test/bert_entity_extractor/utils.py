import unicodedata as ud


def normalize(text):
    """Hàm chuẩn hóa dữ liệu, xóa các khoảng trắng thừa đi."""
    return normalize_unicode(" ".join(text.split()))


def normalize_unicode(text):
    """Chuẩn hóa unicode cho dữ liệu text."""
    return ud.normalize("NFC", text)