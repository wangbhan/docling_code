"""
DOCX 转 Markdown 脚本

功能：
1. 使用 docling 将 DOCX 文档转换为 Markdown
2. 使用本地 RapidOCR 模型识别图片中的文字
3. 将图片转换为 base64 编码并嵌入 Markdown
4. OCR 识别的文字作为图片的 alt 文本
5. 保留所有内容（文字、表格、标题等）

输出格式：[OCR识别文字](data:image/png;base64,编码内容)
"""

import os
import re
from typing import Optional

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem


class DocxToMarkdownConverter:
    """DOCX 转 Markdown 转换器，支持 OCR 和图片嵌入"""

    def __init__(
        self,
        ocr_model_path: str = "./demo_models/RapidAI/RapidOCR",
        use_mobile_model: bool = True,
        use_cuda: bool = False,
    ):
        """
        初始化转换器

        Args:
            ocr_model_path: OCR 模型路径
            use_mobile_model: 是否使用 mobile 模型（内存占用更小）
            use_cuda: 是否使用 GPU 加速（需要安装 onnxruntime-gpu 和 CUDA 版 PyTorch）
        """
        self.ocr_model_path = ocr_model_path
        self.use_mobile_model = use_mobile_model
        self.use_cuda = use_cuda

        # 构建模型路径
        self.det_model_path, self.rec_model_path, self.cls_model_path = (
            self._get_model_paths(use_mobile_model)
        )

        # 创建 DocumentConverter 和 OCR 引擎
        self.converter = self._create_converter()
        self.ocr_engine = self._create_ocr_engine()

    def _get_model_paths(self, use_mobile_model: bool) -> tuple[str, str, str]:
        """获取 OCR 模型路径"""
        if use_mobile_model:
            det_model = "ch_PP-OCRv5_mobile_det.onnx"
            rec_model = "ch_PP-OCRv5_rec_mobile_infer.onnx"
        else:
            det_model = "ch_PP-OCRv5_server_det.onnx"
            rec_model = "ch_PP-OCRv5_rec_server_infer.onnx"

        det_model_path = os.path.join(
            self.ocr_model_path,
            "onnx",
            "PP-OCRv5",
            "det",
            det_model,
        )
        rec_model_path = os.path.join(
            self.ocr_model_path,
            "onnx",
            "PP-OCRv5",
            "rec",
            rec_model,
        )
        cls_model_path = os.path.join(
            self.ocr_model_path,
            "onnx",
            "PP-OCRv4",
            "cls",
            "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        )

        # 验证模型文件是否存在
        for path in [det_model_path, rec_model_path, cls_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"OCR 模型文件不存在: {path}")

        print(f"检测模型: {det_model_path}")
        print(f"识别模型: {rec_model_path}")
        print(f"分类模型: {cls_model_path}")

        return det_model_path, rec_model_path, cls_model_path

    def _create_ocr_engine(self) -> RapidOCR:
        """创建 RapidOCR 引擎用于图片 OCR"""
        return RapidOCR(
            det_model_path=self.det_model_path,
            rec_model_path=self.rec_model_path,
            cls_model_path=self.cls_model_path,
            use_cuda=self.use_cuda,  # GPU 加速
        )

    def _create_converter(self) -> DocumentConverter:
        """创建 DocumentConverter 实例"""
        # 配置 OCR 选项
        ocr_options = RapidOcrOptions(
            det_model_path=self.det_model_path,
            rec_model_path=self.rec_model_path,
            cls_model_path=self.cls_model_path,
        )

        # 配置 GPU 加速选项
        accelerator_options = AcceleratorOptions()
        if self.use_cuda:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA
            )
            print("已启用 GPU 加速 (CUDA)")

        # 配置 Pipeline 选项（包含 accelerator_options）
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_options,
            do_ocr=True,
            do_table_structure=True,
            generate_picture_images=True,  # 生成图片数据，用于后续 OCR
            accelerator_options=accelerator_options,  # GPU 加速配置
        )

        # 创建 DocumentConverter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )

        return converter

    def _extract_ocr_text_from_pictures(self, doc) -> list[str]:
        """
        使用 RapidOCR 对图片进行 OCR 识别

        Args:
            doc: DoclingDocument 对象

        Returns:
            OCR 识别文字列表，按图片出现顺序排列
        """
        ocr_texts = []

        # 遍历文档中的所有图片
        for item, _ in doc.iterate_items():
            if isinstance(item, PictureItem):
                # 获取 PIL Image 对象
                pil_image = item.get_image(doc)

                if pil_image is None:
                    print(f"警告: 无法获取图片数据")
                    ocr_texts.append("")
                    continue

                # 将 PIL Image 转换为 numpy 数组供 RapidOCR 使用
                img_array = np.array(pil_image)

                # 使用 RapidOCR 进行识别
                result, _ = self.ocr_engine(img_array)
                print(f"result:{result}")
                if result:
                    # 合并所有识别的文字
                    text = " ".join([line[1] for line in result])
                    ocr_texts.append(text.strip())
                    print(f"OCR 识别结果: {text.strip()[:50]}...")
                else:
                    ocr_texts.append("")

        return ocr_texts

    def _post_process_markdown(
        self, markdown_content: str, ocr_texts: list[str]
    ) -> str:
        """
        后处理 Markdown 内容，将 OCR 文字添加为 alt 文本

        Args:
            markdown_content: 原始 Markdown 内容
            ocr_texts: OCR 识别的文字列表

        Returns:
            处理后的 Markdown 内容
        """
        # 使用索引跟踪当前处理的图片
        ocr_index = [0]

        # 正则匹配 base64 图片：![](data:image/...;base64,...) 或 ![...](data:image/...;base64,...)
        pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"

        def replace_image(match):
            current_alt = match.group(1)
            image_type = match.group(2)
            base64_data = match.group(3)

            # 获取对应的 OCR 文字
            alt_text = "图片"
            if ocr_index[0] < len(ocr_texts) and ocr_texts[ocr_index[0]]:
                alt_text = ocr_texts[ocr_index[0]]
            elif current_alt:
                alt_text = current_alt

            ocr_index[0] += 1

            # 返回带 alt 文本的图片标记
            return f"![{alt_text}](data:image/{image_type};base64,{base64_data})"

        # 替换所有图片
        processed_content = re.sub(pattern, replace_image, markdown_content)

        return processed_content

    def convert(
        self,
        docx_path: str,
        output_path: Optional[str] = None,
        raise_on_ocr_error: bool = True,
    ) -> str:
        """
        转换 DOCX 文档为 Markdown

        Args:
            docx_path: DOCX 文档路径
            output_path: 输出 Markdown 文件路径（可选）
            raise_on_ocr_error: OCR 失败时是否抛出错误

        Returns:
            Markdown 内容字符串

        Raises:
            FileNotFoundError: 输入文件不存在
            RuntimeError: OCR 识别失败（当 raise_on_ocr_error=True 时）
        """
        # 验证输入文件
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"输入文件不存在: {docx_path}")

        print(f"开始转换: {docx_path}")

        # 1. 转换文档
        print("正在转换文档...")
        conversion_result = self.converter.convert(source=docx_path)

        # 2. 获取文档对象
        doc = conversion_result.document

        # 3. 提取 OCR 文字
        print("正在提取 OCR 识别的文字...")
        ocr_texts = self._extract_ocr_text_from_pictures(doc)

        # 检查是否有图片但没有 OCR 文字
        has_pictures = any(
            isinstance(item, PictureItem) for item, _ in doc.iterate_items()
        )

        if has_pictures and not any(ocr_texts) and raise_on_ocr_error:
            raise RuntimeError("文档中包含图片但 OCR 识别失败，未能提取文字")

        print(f"提取到 {len([t for t in ocr_texts if t])} 个图片的 OCR 文字")

        # 4. 导出为 Markdown（嵌入 base64 图片）
        print("正在导出 Markdown...")
        markdown_content = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

        # 5. 后处理：将 OCR 文字添加为 alt 文本
        print("正在处理图片 alt 文本...")
        markdown_content = self._post_process_markdown(markdown_content, ocr_texts)

        # 6. 保存到文件（如果指定了输出路径）
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"转换完成，已保存到: {output_path}")

        return markdown_content


def main():
    """主函数示例"""

    # 创建转换器实例
    converter = DocxToMarkdownConverter(
        ocr_model_path="./demo_models/RapidAI/RapidOCR",
        use_mobile_model=True,
    )

    # 转换文档
    try:
        markdown_content = converter.convert(
            docx_path="./demo4.docx",  # 输入 DOCX 文件
            output_path="./docx_output.md",  # 输出 Markdown 文件
            raise_on_ocr_error=True,
        )

        print("\n转换成功！")
        print(f"Markdown 内容长度: {len(markdown_content)} 字符")

    except FileNotFoundError as e:
        print(f"错误：{e}")
    except RuntimeError as e:
        print(f"OCR 错误：{e}")
    except Exception as e:
        print(f"未知错误：{e}")


if __name__ == "__main__":
    main()
