import random

import gradio as gr
from langchain_community.callbacks import get_openai_callback
from typing import List, Dict
import json
import asyncio

from ktem.pages.src.mcqgenerator.utils import read_file,get_table_data
from ktem.pages.src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from ktem.pages.src.mcqgenerator.RAG import RAG
from ktem.pages.src.mcqgenerator.constants import type_of_question_info


class GradioMCQPage:
    def __init__(self):
        self.review = ""  # 初始化 review 为空字符串
        self.show_review_md = None  # 添加一个新的 Markdown 组件来显示 review
        self.quiz_data = {}
        self.edit_mode = False
        self.TYPE = ["多项选择题", "单项选择题", "对错题", "填空题"]

    def ui(self):
        with gr.Blocks() as self.mcq_interface:
            gr.Markdown("# MCQ Generator")
            with gr.Tab("题目生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        self.file_upload = gr.File(label="上传 PDF、RTF 或 TXT 文件")
                        self.text_input = gr.Textbox(label="或在此输入文本", lines=5, value="神经网络的发展历程")
                        with gr.Row():
                            self.mcq_count = gr.Slider(minimum=3, maximum=10, value=3, step=1, label="生成题目数量")
                            self.subject = gr.Textbox(label="科目", value="深度学习")
                        with gr.Row():
                            self.tone = gr.Textbox(label="问题复杂度", placeholder="简单", value="简单")
                            self.question_type = gr.Dropdown(choices=self.TYPE, label="题目类型", value="单项选择题")
                        # 移除了不需要的选项
                        self.generate_btn = gr.Button("生成题目")
                        self.start_study_btn = gr.Button("开始学习")

                    with gr.Column(scale=2):
                        # 单独定义一个 Markdown 区域用于显示
                        self.output_area_md = gr.Markdown("生成的题目将显示在这里", visible=True)
                        # 编辑模式下的 Textbox 列表，初始隐藏
                        self.edit_textboxes = [
                            gr.Textbox(visible=False,interactive=True,label=None) for _ in range(40)  # 根据需要调整数量
                        ]

                        self.edit_mode_checkbox = gr.Checkbox(label="编辑模式", value=False)
                        self.save_btn = gr.Button("保存修改")
                        self.export_btn = gr.Button("导出题目")
                        self.show_review = gr.Accordion("Review", open=False)
                        with self.show_review:
                            self.show_review_md = gr.Markdown(self.review, visible=True)
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

            # 设置刷新按钮的回调
            self.refresh_btn.click(
                fn=self.refresh_mistake_tab,
                inputs=[self.question_type],
                outputs=[item for sublist in self.mistake_question_block for item in sublist]  # 展开所有问题和选择的组件
            )

            self.generate_btn.click(
                fn=self.generate_mcqs,
                inputs=[
                    self.file_upload,
                    self.text_input,
                    self.mcq_count,
                    self.subject,
                    self.tone,
                    self.question_type
                ],
                outputs=[self.output_area_md, self.show_review_md]+[item for sublist in self.question_block for item in sublist]  # 添加 show_review_md 作为输出
            )

            self.edit_mode_checkbox.change(
                fn=self.toggle_edit_mode,
                inputs=[self.edit_mode_checkbox],
                outputs=[self.output_area_md] + self.edit_textboxes
            )

            success_message = gr.HTML()
            self.save_btn.click(
                fn=self.save_quiz_data,
                inputs=self.edit_textboxes,
                outputs=success_message
            )
            self.export_btn.click(
                fn=self.export_quiz,
                outputs=gr.File(visible=False)
            )
            self.start_study_btn.click(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=study_tab
            )
            self.submit_btn.click(
                fn=self.evaluate_answers,
                inputs=[self.question_type] + [opt[1] for opt in self.question_block],
                outputs=self.score_display
            )

    async def generate_mcqs(self, file, text, mcq_count, subject, tone, question_type):
        try:


            self.quiz_data = {
                        "1": {
                            "question": "下列哪个选项最准确地描述了连接主义（Connectionism）？",
                            "options": {
                                "A": "连接主义认为认知过程可以通过明确的符号系统进行推理，类似于传统的符号主义。",
                                "B": "连接主义专注于将智能归结为生物神经元网络的简化模型，模拟生物学习过程。",
                                "C": "连接主义模型由大量简单的处理单元组成的网络，能够进行分布式信息处理和自适应学习。",
                                "D": "连接主义认为智能的核心在于符号计算，并通过逻辑推理实现复杂的任务。"
                            },
                            "correct": "C",
                            "explanation": "连接主义的核心在于模拟生物神经网络的工作机制，使用由大量简单单元组成的网络进行非线性和分布式的处理。选项 c 准确描述了连接主义的特征，而其他选项更多反映了符号主义或对连接主义的误解。"
                        },
                        "2": {
                            "question": "关于表示学习（Representation Learning），以下哪种说法是正确的？",
                            "options": {
                                "A": "表示学习的核心是从数据中提取特征，而不必关注特征的语义意义。",
                                "B": "表示学习的主要挑战是从低级特征中自动学习到可以解决高级任务的语义表示。",
                                "C": "局部表示通常优于分布式表示，因为局部表示更易于进行模型解释。",
                                "D": "表示学习主要依赖于通过手工设计特征来提高模型性能。"
                            },
                            "correct": "B",
                            "explanation": "表示学习的目标是自动发现适合于特定任务的有效表示形式，尤其是通过从原始数据中学习出具有语义含义的高层特征。解决“语义鸿沟”问题是表示学习的关键挑战，因此选项 b 是正确的，而其他选项则或忽略了自动学习的能力，或对表示学习的目的有所误解。"
                        },
                        "3": {
                            "question": "在人工智能的发展历程中，以下哪个时期被认为是专注于通过构建知识库来提升智能系统的？",
                            "options": {
                                "A": "推理期，该时期通过符号推理来模拟人类逻辑能力，忽略了对知识的积累。",
                                "B": "学习期，强调机器从数据中自动学习，但尚未发展知识库的概念。",
                                "C": "知识期，此阶段强调了知识对于构建智能系统的重要性，专家系统便是在此基础上诞生的。",
                                "D": "智能期，该时期机器开始通过大数据和深度学习模型来构建复杂的知识系统。"
                            },
                            "correct": "C",
                            "explanation": "知识期是人工智能发展的重要阶段，研究者认识到构建知识库的重要性，并致力于将领域知识融入智能系统，以提升其决策和推理能力。专家系统是该时期的代表性成果，选项 c 正确地捕捉了这一关键发展。"
                        }
                    }
            self.review="nihao"
            question_updates = self.question_show_tab(question_type, self.quiz_data)
            return [self.format_quiz_for_display(self.quiz_data), gr.update(value=self.review)] + question_updates

        except Exception as e:

            return f"发生错误：{str(e)}"

    def toggle_edit_mode(self, edit_mode):
        if edit_mode:
            # 创建 quiz_data 的副本
            formatted_text = self.format_quiz_for_edit()
            updates = [gr.update(visible=False)]  # 隐藏 Markdown
            for i, (textbox, content) in enumerate(zip(self.edit_textboxes, formatted_text)):
                updates.append(gr.update(visible=True, value=content,label=None))
            for i in range(len(formatted_text), len(self.edit_textboxes)):
                updates.append(gr.update(visible=False))
        else:
            # 返回到 Markdown 显示模式
            updates = [gr.update(visible=True, value=self.format_quiz_for_display(self.quiz_data))]
            updates += [gr.update(visible=False) for _ in self.edit_textboxes]
        return updates

    def format_quiz_for_display(self, quiz_data: Dict) -> str:
        formatted_output = ""
        for idx, (_, question) in enumerate(quiz_data.items(), start=1):
            formatted_output += f"### 问题 {idx}：{question['question']}\n\n"
            for opt, text in question['options'].items():
                formatted_output += f"- {opt}：{text}\n"
            formatted_output += f"\n正确答案：{question['correct']}\n"
            formatted_output += f"解释：{question['explanation']}\n\n"
        return formatted_output

    def format_quiz_for_edit(self):
        output = []
        for idx, (_, question) in enumerate(self.quiz_data.items(), start=1):
            output.append(f"问题 {idx}：{question['question']}")
            for opt, text in question['options'].items():
                output.append(f"{opt}：{text}")
            output.append(f"正确答案：{question['correct']}")
            output.append(f"解释：{question['explanation']}")
        return output

    def save_quiz_data(self, *inputs):
        edit_textbox_index = 0  # 用于遍历 self.edit_textboxes 的索引

        # 遍历 self.quiz_data 中的每一道题目
        for question_id, question_data in self.quiz_data.items():
            # 更新问题文本
            question_data["question"] = inputs[edit_textbox_index].split('：', 1)[1]
            edit_textbox_index += 1

            # 更新选项文本
            for option_key in question_data["options"]:
                question_data["options"][option_key] = inputs[edit_textbox_index].split('：', 1)[1]
                edit_textbox_index += 1

            # 更新正确答案
            question_data["correct"] = inputs[edit_textbox_index].split('：', 1)[1]
            edit_textbox_index += 1

            # 更新解释文本
            question_data["explanation"] = inputs[edit_textbox_index].split('：', 1)[1]
            edit_textbox_index += 1

        # 更新后保存修改到 quiz_data
        # return gr.update(value="修改已保存")

        with open("quiz_results.json", "w", encoding="utf-8") as f:
            json.dump(self.quiz_data, f, ensure_ascii=False, indent=4)
        return "<p style='color: green;'>结果已保存</p>"
    def export_quiz(self):
        # 将测验数据保存到文件
        with open("quiz_results.json", "w", encoding="utf-8") as f:
            json.dump(self.quiz_data, f, ensure_ascii=False, indent=4)

        # 返回文件对象
        return gr.File(value="quiz_results.json", visible=True)
    def create_question_block(self, idx):
        question_text = gr.Markdown(visible=False)
        radio_buttons = gr.Radio(choices=[], visible=False)
        checkbox_buttons = gr.CheckboxGroup(choices=[], visible=False)
        return question_text, radio_buttons, checkbox_buttons

    def evaluate_answers(self, question_type, *answers):
        correct_count = 0
        mistake_book = {"mcq": {}, "other": {}}  # 错题集

        # 读取现有的错题集
        try:
            with open('../Mistake Book.json', 'r', encoding='utf-8') as M_file:
                MISTAKE_JSON = json.load(M_file)
        except (FileNotFoundError, json.JSONDecodeError):
            MISTAKE_JSON = {"mcq": {}, "other": {}}

        # 获取当前错题集的最后一个索引
        if MISTAKE_JSON["mcq"]:
            last_mcq_index = max(int(k) for k in MISTAKE_JSON["mcq"].keys())
        else:
            last_mcq_index = 0

        if MISTAKE_JSON["other"]:
            last_other_index = max(int(k) for k in MISTAKE_JSON["other"].keys())
        else:
            last_other_index = 0

        def is_question_in_mistake_book(question, mistake_book, qtype):
            """检查题目是否已经在错题集中"""
            for _, data in mistake_book[qtype].items():
                if data["question"] == question["question"]:
                    return True
            return False

        for idx, answer in enumerate(answers):
            if answer is None:
                break

            # 获取正确答案
            correct = self.quiz_data[str(idx + 1)]["correct"].replace('，', ',').split(",")
            correct_options = [option.strip().upper() for option in correct if option.strip()]

            if question_type == "多项选择题":
                # 多选题的处理
                if isinstance(answer, list) and len(answer) >= 2:
                    user_options = [opt.split(":")[0].strip().upper() for opt in answer]
                    correct_options_set = set(correct_options)
                    user_options_set = set(user_options)

                    if user_options_set == correct_options_set:
                        # 全部选对得满分
                        correct_count += 1
                    else:
                        # 少选、多选、错选均不得分,加入多选错题集
                        if not is_question_in_mistake_book(self.quiz_data[str(idx + 1)], MISTAKE_JSON, "mcq"):
                            last_mcq_index += 1
                            mistake_book["mcq"][str(last_mcq_index)] = self.quiz_data[str(idx + 1)]
                else:
                    # 不满足至少选择两个选项的要求,加入多选错题集
                    if not is_question_in_mistake_book(self.quiz_data[str(idx + 1)], MISTAKE_JSON, "mcq"):
                        last_mcq_index += 1
                        mistake_book["mcq"][str(last_mcq_index)] = self.quiz_data[str(idx + 1)]
            elif question_type == "单项选择题":
                # 单选题的处理
                if isinstance(answer, str):
                    user_option = answer.split(":")[0].strip().upper()
                    if user_option in correct_options:
                        correct_count += 1
                    else:
                        # 错选,加入单选错题集
                        if not is_question_in_mistake_book(self.quiz_data[str(idx + 1)], MISTAKE_JSON, "other"):
                            last_other_index += 1
                            mistake_book["other"][str(last_other_index)] = self.quiz_data[str(idx + 1)]
            else:
                raise ValueError(f"未知的题目类型: {question_type}")

        # 更新现有的错题集
        MISTAKE_JSON["mcq"].update(mistake_book["mcq"])
        MISTAKE_JSON["other"].update(mistake_book["other"])

        # 将更新后的错题集写回文件
        with open('../Mistake Book.json', 'w', encoding='utf-8') as M_file:
            json.dump(MISTAKE_JSON, M_file, ensure_ascii=False, indent=4)

        if question_type == "多项选择题":
            return f"评分规则：全选得满分，少选、多选、错选均不得分\n\n你的得分是: {correct_count} / {len(self.quiz_data)}"
        else:
            return f"评分规则：选对得分\n\n你的得分是: {correct_count} / {len(self.quiz_data)}"

    def refresh_mistake_tab(self, question_type):
        forgotten_questions = self.get_random_questions()
        if question_type == "多项选择题":
            return self.question_show_tab(question_type, forgotten_questions[0])
        return self.question_show_tab(question_type, forgotten_questions[1])

    def get_random_questions(self, num_mcq=5, num_other=5):
        try:
            with open('../Mistake Book.json', 'r', encoding='utf-8') as M_file:
                mistake_book = json.load(M_file)
        except (FileNotFoundError, json.JSONDecodeError):
            mistake_book = {"mcq": {}, "other": {}}

        # 分别从 mcq 和 other 中随机抽取指定数量的题目
        selected_mcq = {}
        selected_other = {}

        mcq_keys = list(mistake_book["mcq"].keys())
        if len(mcq_keys) < num_mcq:
            num_mcq = len(mcq_keys)
        selected_mcq_keys = random.sample(mcq_keys, num_mcq)
        for key in selected_mcq_keys:
            selected_mcq[key] = mistake_book["mcq"][key]

        other_keys = list(mistake_book["other"].keys())
        if len(other_keys) < num_other:
            num_other = len(other_keys)
        selected_other_keys = random.sample(other_keys, num_other)
        for key in selected_other_keys:
            selected_other[key] = mistake_book["other"][key]

        return selected_mcq, selected_other

    def question_show_tab(self, question_type, forgotten_questions):
        question_updates = []
        for idx, data in enumerate(forgotten_questions.values()):
            question_display = f"### 问题 {idx + 1}: {data['question']}"
            choices = [f"{key}: {value}" for key, value in data['options'].items()]
            if question_type == "多项选择题":
                # 显示 CheckboxGroup 组件，隐藏 Radio 组件
                question_updates.extend([
                    gr.update(value=question_display, visible=True),
                    gr.update(visible=False),  # Radio 组件
                    gr.update(choices=choices, visible=True, interactive=True)  # CheckboxGroup 组件
                ])
            else:
                # 显示 Radio 组件，隐藏 CheckboxGroup 组件
                question_updates.extend([
                    gr.update(value=question_display, visible=True),
                    gr.update(choices=choices, visible=True, interactive=True),  # Radio 组件
                    gr.update(visible=False)  # CheckboxGroup 组件
                ])

            # 隐藏多余的问题块
        for _ in range(len(forgotten_questions), len(self.question_block) * 3):
            question_updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])

        return question_updates

mcq_generator = GradioMCQPage()
mcq_generator.ui()
mcq_generator.mcq_interface.launch(
    server_name="172.18.232.176",
    share=True,
    inbrowser=False,
)