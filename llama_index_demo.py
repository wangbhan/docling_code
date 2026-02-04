# import os
# from typing import List, Optional
# from pathlib import Path
#
# from modelscope import snapshot_download
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from llama_index.core import Document
# from llama_index.core.readers.base import BaseReader
#
#
# class DoclingReader(BaseReader):
#     """Docling document reader with custom OCR configuration."""
#
#     def __init__(
#             self,
#             doc_converter: Optional[DocumentConverter] = None,
#             download_path: str = "./demo_models/RapidAI/RapidOCR",
#             use_postprocessor: bool = True,
#     ):
#         """
#         初始化 DoclingReader
#
#         Args:
#             doc_converter: 自定义的 DocumentConverter 实例
#             download_path: OCR 模型路径
#             use_postprocessor: 是否使用后处理器
#         """
#         self.use_postprocessor = use_postprocessor
#
#         if doc_converter is not None:
#             self.converter = doc_converter
#         else:
#             # 使用默认配置创建 converter
#             det_model_path = os.path.join(
#                 download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_mobile_det.onnx"
#             )
#             rec_model_path = os.path.join(
#                 download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_mobile_infer.onnx"
#             )
#             cls_model_path = os.path.join(
#                 download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
#             )
#
#             ocr_options = RapidOcrOptions(
#                 det_model_path=det_model_path,
#                 rec_model_path=rec_model_path,
#                 cls_model_path=cls_model_path,
#             )
#
#             pipeline_options = PdfPipelineOptions(
#                 ocr_options=ocr_options,
#             )
#
#             self.converter = DocumentConverter(
#                 format_options={
#                     InputFormat.PDF: PdfFormatOption(
#                         pipeline_options=pipeline_options,
#                     ),
#                 },
#             )
#
#     def load_data(
#             self,
#             file_path: str,
#             extra_info: Optional[dict] = None,
#     ) -> List[Document]:
#         """
#         加载并转换文档
#
#         Args:
#             file_path: 文档路径
#             extra_info: 额外的元数据信息
#
#         Returns:
#             Document 对象列表
#         """
#         # 转换文档
#         conversion_result = self.converter.convert(source=file_path)
#         # 应用后处理(如果启用)
#         if self.use_postprocessor:
#             try:
#                 from hierarchical.postprocessor import ResultPostprocessor
#                 ResultPostprocessor(conversion_result).process()
#             except ImportError:
#                 print("Warning: ResultPostprocessor not available, skipping postprocessing")
#
#         # 导出为 Markdown
#         doc = conversion_result.document
#         md_content = doc.export_to_markdown()
#
#         # 构建元数据
#         metadata = {
#             "file_path": file_path,
#             "file_name": Path(file_path).name,
#             "file_type": "pdf",
#         }
#
#         if extra_info:
#             metadata.update(extra_info)
#
#         # 创建 Document 对象
#         document = Document(
#             text=md_content,
#             metadata=metadata,
#         )
#
#         # 返回列表(符合 LlamaIndex Reader 标准)
#         return [document]
#
#
# def main():
#
#     # 方式2: 使用自定义 converter(与你原来的代码一致)
#     download_path = "./demo_models/RapidAI/RapidOCR"
#     det_model_path = os.path.join(
#         download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_mobile_det.onnx"
#     )
#     rec_model_path = os.path.join(
#         download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_mobile_infer.onnx"
#     )
#     cls_model_path = os.path.join(
#         download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
#     )
#
#     ocr_options = RapidOcrOptions(
#         det_model_path=det_model_path,
#         rec_model_path=rec_model_path,
#         cls_model_path=cls_model_path,
#     )
#
#     pipeline_options = PdfPipelineOptions(
#         ocr_options=ocr_options,
#     )
#
#     converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(
#                 pipeline_options=pipeline_options,
#             ),
#         },
#     )
#
#     reader = DoclingReader(doc_converter=converter)
#     documents = reader.load_data("../demo4.pdf")
#
#     # 打印结果
#     print(f"Loaded {len(documents)} document(s)")
#     print(documents)
#     print(documents[0])
#
#     from llama_index.core import Document
#     documents = Document(text=documents[0].text_resource.text, doc_id=documents[0].doc_id)
#
#
#
# if __name__ == "__main__":
#     main()