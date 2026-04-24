from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from image_processing import (
    ImageProcessingError,
    apply_convolution,
    apply_histogram_specification,
    create_histogram_preview,
    ensure_odd,
)


def _decode_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError("Gagal membaca file gambar.")
    return image


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _process_image(
    original_image: np.ndarray,
    reference_image: np.ndarray | None,
    filter_name: str,
    kernel_size: int,
    strength: int,
    use_hist_spec: bool,
    histogram_mode: str,
) -> np.ndarray:
    working = original_image.copy()
    working = apply_convolution(working, filter_name, kernel_size, strength)

    if use_hist_spec:
        if reference_image is None:
            raise ImageProcessingError(
                "Histogram specification aktif, tetapi gambar referensi belum di-upload."
            )
        working = apply_histogram_specification(working, reference_image)

    if histogram_mode == "Grayscale":
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        working = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return working


def main() -> None:
    st.set_page_config(page_title="Peningkatan Mutu Citra", layout="wide")
    st.title("Peningkatan Mutu Citra")
    st.write("Metode: Histogram Specification dan Convolution (Mask Processing).")

    with st.sidebar:
        st.header("Kontrol Peningkatan Citra")
        filter_name = st.selectbox(
            "Convolution (Mask Processing)",
            ["None", "Smoothing", "Sharpening", "Sobel Edge", "Laplacian Edge"],
        )

        kernel_size = st.slider("Kernel Size", min_value=1, max_value=15, value=3, step=1)
        kernel_size = ensure_odd(kernel_size, minimum=1)

        strength = st.slider("Sharpen Strength", min_value=0, max_value=30, value=5, step=1)
        use_hist_spec = st.toggle("Aktifkan Histogram Specification", value=False)
        histogram_mode = st.selectbox("Mode Tampilan Histogram", ["RGB", "Grayscale"])

    st.header("Input / Output")
    col_upload_1, col_upload_2 = st.columns(2)
    with col_upload_1:
        input_file = st.file_uploader(
            "Upload Gambar",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )
    with col_upload_2:
        reference_file = st.file_uploader(
            "Upload Gambar Referensi Histogram (opsional)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )

    if input_file is None:
        st.info("Silakan upload gambar daun untuk mulai memproses.")
    else:
        try:
            original_image = _decode_upload(input_file)
            reference_image = _decode_upload(reference_file) if reference_file is not None else None

            processed_image = _process_image(
                original_image=original_image,
                reference_image=reference_image,
                filter_name=filter_name,
                kernel_size=kernel_size,
                strength=strength,
                use_hist_spec=use_hist_spec,
                histogram_mode=histogram_mode,
            )

            before_hist = create_histogram_preview(original_image, mode=histogram_mode)
            after_hist = create_histogram_preview(processed_image, mode=histogram_mode)

            left, right = st.columns(2)
            with left:
                st.subheader("Sebelum")
                st.image(_to_rgb(original_image), channels="RGB", use_container_width=True)
                st.caption("Histogram Sebelum")
                st.image(_to_rgb(before_hist), channels="RGB", use_container_width=True)
            with right:
                st.subheader("Sesudah")
                st.image(_to_rgb(processed_image), channels="RGB", use_container_width=True)
                st.caption("Histogram Sesudah")
                st.image(_to_rgb(after_hist), channels="RGB", use_container_width=True)

            success, encoded = cv2.imencode(".png", processed_image)
            if success:
                st.download_button(
                    label="Simpan Hasil (PNG)",
                    data=encoded.tobytes(),
                    file_name="hasil_pengolahan_daun.png",
                    mime="image/png",
                )
        except ImageProcessingError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Terjadi error saat memproses gambar: {exc}")

if __name__ == "__main__":
    main()
