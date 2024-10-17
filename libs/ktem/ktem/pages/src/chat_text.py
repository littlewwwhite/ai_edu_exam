import hashlib
import random

import gradio as gr
import json

class GradioMCQPage:
    def __init__(self):
        self.quiz_data = {}
        self.user_answers = {}
        self.TYPE = ["多项选择题", "单项选择题", "对错题", "填空题"]

    def ui(self):
        with gr.Blocks() as self.mcq_interface:
            gr.Markdown("# MCQ Generator")

            with gr.Tab("题目生成"):
                self.file_upload = gr.File(label="上传文件")
                self.text_input = gr.Textbox(label="输入文本", lines=5, value="神经网络的发展历程")
                self.mcq_count = gr.Slider(minimum=3, maximum=10, value=3, step=1, label="生成题目数量")
                self.subject = gr.Textbox(label="科目", value="深度学习")
                self.tone = gr.Textbox(label="问题复杂度", value="简单")
                self.question_type = gr.Dropdown(choices=self.TYPE, label="题目类型")
                self.generate_btn = gr.Button("生成题目")
                self.start_study_btn = gr.Button("开始学习")

            with gr.Tab("学习界面", visible=False) as study_tab:
                self.question_block = [self.create_question_block(i) for i in range(20)]
                self.submit_btn = gr.Button("提交答案")
                self.score_display = gr.Markdown("")

            with gr.Tab("错题温习", visible=True) as mistake_tab:
                self.mistake_question_block = [self.create_question_block(i) for i in range(30)]
                self.submit_btn_ = gr.Button("提交答案")
                self.score_display_ = gr.Markdown("")

                # 添加刷新错题温习界面的按钮
                self.refresh_btn = gr.Button("刷新错题")
            with gr.Tab(label="对话学习", visible=True) as chat_tab:
                self.chatbot = gr.Chatbot(
                    placeholder=(
                        "This is the beginning of a new conversation.\nIf you are new, "
                        "visit the Help tab for quick instructions."
                    ),
                    show_label=False,
                    elem_id="main-chat-bot",
                    show_copy_button=True,
                    likeable=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    self.text_input = gr.Text(
                        placeholder="Chat input",
                        scale=15,
                        container=False,
                        max_lines=10,
                    )
                    self.submit_btn = gr.Button(
                        value="Send",
                        scale=1,
                        min_width=10,
                        variant="primary",
                        elem_classes=["cap-button-height"],
                    )
                    self.regen_btn = gr.Button(
                        value="Regen",
                        scale=1,
                        min_width=10,
                        elem_classes=["cap-button-height"],
                    )


mcq_generator = GradioMCQPage()
mcq_generator.ui()
mcq_generator.mcq_interface.launch(
    server_name="172.18.232.176",
    server_port=7862,
    share=True,
    inbrowser=False,
)
