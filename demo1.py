from llama_index.readers.docling import DoclingReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat
from modelscope import snapshot_download

import os
# 配置使用本地模型
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

# 指定本地模型路径
download_path = snapshot_download(repo_id="RapidAI/RapidOCR", cache_dir="demo_models")
# 文本检测
det_model_path = os.path.join(
    download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_server_det.onnx"
)
# 文本识别
rec_model_path = os.path.join(
    download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
)
# 文本方向分类 / 文本角度矫正
cls_model_path = os.path.join(
    download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
)

ocr_options = RapidOcrOptions(
    det_model_path=det_model_path,
    rec_model_path=rec_model_path,
    cls_model_path=cls_model_path,
)

pipeline_options = PdfPipelineOptions(
    ocr_options=ocr_options,
)

# 创建 DocumentConverter 并传入配置
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        ),
    },
)

# 使用自定义的 converter 创建 reader
reader = DoclingReader(doc_converter=converter)

# 加载文档
documents = reader.load_data("../demo4.pdf")
print(documents)

print(documents[0].text_resource.text)