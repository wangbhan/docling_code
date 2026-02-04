
import os

from modelscope import snapshot_download

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    # Source document to convert
    source = "../demo4.pdf"

    download_path = "./demo_models/RapidAI/RapidOCR"

    # 文本检测 内存21.2
    det_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_mobile_det.onnx"
    )
    # 文本识别
    rec_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_mobile_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    )
    print(det_model_path)
    print(rec_model_path)
    print(cls_model_path)
    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    # Convert the document

    from docling.document_converter import DocumentConverter
    from hierarchical.postprocessor import ResultPostprocessor

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )


    conversion_result = converter.convert(source=source)
    # 添加后处理
    ResultPostprocessor(conversion_result).process()

    doc = conversion_result.document
    md = doc.export_to_markdown()
    with open("demo5.md", "w", encoding="utf-8") as f:
        f.write(md)

    from llama_index.core import Document
    documents = Document(text=md)
    print(documents)


if __name__ == "__main__":
    main()