import random
import os
import gradio as gr
from langchain_community.callbacks import get_openai_callback
from typing import List, Dict
import json
import asyncio

from ktem.pages.src.mcqgenerator.utils import read_file,get_table_data
from ktem.pages.src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from ktem.pages.src.mcqgenerator.RAG import RAG,generate_response
from ktem.pages.src.mcqgenerator.constants import type_of_question_info,mistake_book_dir,sys_pro
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory


class GradioMCQPage:
    def __init__(self, app):
        self._app = app
        self.review = ""  # 初始化 review 为空字符串
        self.show_review_md = None  # 添加一个新的 Markdown 组件来显示 review
        self.quiz_data = {}
        self.mistake_book_quiz = []
        self.information=[]
        self.edit_mode = False
        self.TYPE = ["多项选择题", "单项选择题", "对错题"]
        self.level=["简单","适中","困难","创新"]
        self.count=0


    def ui(self):
        with gr.Blocks() as self.mcq_interface:
            gr.Markdown("# Learn&Exam")
            with gr.Tab("题目生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        self.file_upload = gr.File(label="上传 PDF、RTF 或 TXT 文件（临时），长期文档请上传至FIle")
                        self.text_input = gr.Textbox(label="或在此输入文本你想要生成试卷的相关信息", lines=5, value="")
                        with gr.Row():
                            self.mcq_count = gr.Slider(minimum=3, maximum=10, value=3, step=1, label="生成题目数量")
                            self.subject = gr.Textbox(label="科目名称（可模糊）", value="")
                        with gr.Row():
                            self.tone = gr.Dropdown(choices=self.level, label="题目复杂度", value="适中")
                            self.question_type = gr.Dropdown(choices=self.TYPE, label="题目类型", value="单项选择题")
                        # 移除了不需要的选项
                        self.generate_btn = gr.Button("生成题目")
                        # self.start_study_btn = gr.Button("开始学习")

                    with gr.Column(scale=2):
                        # 单独定义一个 Markdown 区域用于显示
                        self.output_area_md = gr.Markdown("生成的题目将显示在这里", visible=True)
                        # 编辑模式下的 Textbox 列表，初始隐藏
                        self.edit_textboxes = [
                            gr.Textbox(visible=False, interactive=True, label=None) for _ in range(40)  # 根据需要调整数量
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
                tab1_identifier = gr.Textbox(value="study", visible=False)

            with gr.Tab("错题温习", visible=True) as mistake_tab:
                self.mistake_question_block = [self.create_question_block(i) for i in range(30)]
                self.submit_btn_ = gr.Button("提交答案")
                self.score_display_ = gr.Markdown("")
                tab2_identifier = gr.Textbox(value="review", visible=False)

                # 添加刷新错题温习界面的按钮
                self.refresh_btn = gr.Button("刷新错题")

            with gr.Tab("错题对话", visible=True) as chat_tab:
                self.chatbot = gr.Chatbot(
                    placeholder="根据错题集对话\n请你回答问题",
                    show_label=False,
                    elem_id="main-chat-bot",
                    show_copy_button=True,
                    likeable=True,
                    bubble_full_width=False,
                    height=500,
                    container=True,
                )
                with gr.Row():
                    self.text_input_chat = gr.Textbox(
                        placeholder="Chat input",
                        scale=15,
                        container=False,
                        max_lines=10,
                    )
                    self.submit_btn_chat = gr.Button(
                        value="Send",
                        scale=1,
                        min_width=10,
                        variant="primary",
                        elem_classes=["cap-button-height"],
                    )

            # 设置刷新按钮的回调
            self.refresh_btn.click(
                fn=self.refresh_mistake_tab,
                inputs=[self.question_type, self._app.user_id],
                outputs=[self.score_display_]+[item for sublist in self.mistake_question_block for item in sublist]  # 展开所有问题和选择的组件
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
                outputs=[self.output_area_md, self.show_review_md, study_tab] + [item for sublist in self.question_block for item
                                                                      in sublist] # 添加 题目、评论和做题显示
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
            # self.start_study_btn.click(
            #     fn=lambda: gr.update(visible=True),
            #     inputs=None,
            #     outputs=study_tab
            # )
            self.submit_btn.click(
                fn=self.evaluate_answers,
                inputs=[self.question_type, tab1_identifier, self._app.user_id] + [opt[1] for opt in self.question_block],
                outputs=self.score_display
            )
            self.submit_btn_.click(
                fn=self.evaluate_answers,
                inputs=[self.question_type, tab2_identifier, self._app.user_id] + [opt[1] for opt in self.mistake_question_block],
                outputs=self.score_display_
            )
            self.text_input_chat.submit(
                self.add_msg,
                [self.text_input_chat, self.chatbot],
                [self.text_input_chat, self.chatbot],
                queue=False).then(
                fn=self.get_response,
                inputs=[self.chatbot, self._app.user_id],
                outputs=[self.chatbot]
            )
            self.submit_btn_chat.click(
                self.add_msg,
                [self.text_input_chat, self.chatbot],
                [self.text_input_chat, self.chatbot],
                queue=False).then(
                fn=self.get_response,
                inputs=[self.chatbot, self._app.user_id],
                outputs=[self.chatbot]
            )
    async def generate_mcqs(self, file, text, mcq_count, subject, tone, question_type):
        try:
            if file:
                text = read_file(file)
            elif not text:
                return "请上传文件或输入文本。"

            # 调用 RAG 函数时，移除了多余的参数
            result = await RAG(question=text, )

            if result["message"] in [
                "I couldn't find any relevant documents to answer your question.",
                "Something went wrong"
            ]:
                result = {"message": text}

            format_ = type_of_question_info[question_type]["format_"]
            examples_json = type_of_question_info[question_type]["examples_json"]
            response_json = type_of_question_info[question_type]["response_json"]

            with open(response_json, 'r') as R_file, open(examples_json, 'r', encoding='utf-8') as E_file:
                RESPONSE_JSON = json.load(R_file)
                EXAMPLES_JSON = json.load(E_file)

            with get_openai_callback() as cb:
                response = generate_evaluate_chain({
                    "text": result["message"],
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON),
                    "examples": json.dumps(EXAMPLES_JSON, ensure_ascii=False),
                    "type_": question_type,
                    "format_": format_
                })
            self.review = response["review"]
            self.quiz_data = response.get("quiz", {})
            self.quiz_data = get_table_data(self.quiz_data)
            print(self.quiz_data)
            print(f"Total Tokens:{cb.total_tokens}")
            print(f"Prompt Tokens:{cb.prompt_tokens}")
            print(f"Completion Tokens:{cb.completion_tokens}")
            print(f"Total Cost:{cb.total_cost}")
            question_updates = self.question_show_tab(question_type, self.quiz_data)
            return [self.format_quiz_for_display(self.quiz_data), gr.update(value=self.review), gr.update(visible=True)] + question_updates

        except Exception as e:
            print(f"Total Tokens:{cb.total_tokens}")
            print(f"Prompt Tokens:{cb.prompt_tokens}")
            print(f"Completion Tokens:{cb.completion_tokens}")
            print(f"Total Cost:{cb.total_cost}")
            return f"发生错误：{str(e)}"

    def toggle_edit_mode(self, edit_mode):
        if edit_mode:
            formatted_text = self.format_quiz_for_edit()
            updates = [gr.update(visible=False)]  # 隐藏 Markdown
            for i, (textbox, content) in enumerate(zip(self.edit_textboxes, formatted_text)):
                updates.append(gr.update(visible=True, value=content, label=None))
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

    def evaluate_answers(self, question_type,study_or_review,User_id, *answers):
        correct_count = 0
        mistake_book = {"mcq": {}, "other": {}}  # 错题集
        result_details = []  # 用于存储所有题目的结果详情

        mistake_book_mapping = {
            "多项选择题": 0,
        }

        # 适应学习界面和温习界面的提交按钮事件
        quiz = self.quiz_data if study_or_review == "study" else self.mistake_book_quiz[
            mistake_book_mapping.get(question_type, 1)]
        quiz_keys=list(quiz.keys())
        # 如果你有更多的 question_type 需要处理，可以在 mistake_book_mapping 中添加更多条目
        # 读取现有的错题集
        try:
            with open(os.path.join(mistake_book_dir, f"Mistake_Book_{User_id}.json"), 'r', encoding='utf-8') as M_file:
                MISTAKE_JSON = json.load(M_file)
        except (FileNotFoundError, json.JSONDecodeError):
            MISTAKE_JSON = {"mcq": {}, "other": {}}

        # 获取当前错题集的最后一个索引
        last_mcq_index = max([int(k) for k in MISTAKE_JSON["mcq"].keys()]) if MISTAKE_JSON["mcq"] else 0
        last_other_index = max([int(k) for k in MISTAKE_JSON["other"].keys()]) if MISTAKE_JSON["other"] else 0

        for idx, answer in enumerate(answers):
            if answer is None:
                break

            question = quiz[quiz_keys[idx]]
            correct = question["correct"].replace('，', ',').split(",")
            correct_options = [option.strip().upper() for option in correct if option.strip()]

            is_correct = False
            if question_type == "多项选择题":
                # 多选题的处理
                if isinstance(answer, list) and len(answer) >= 2:
                    user_options = [opt.split(":")[0].strip().upper() for opt in answer]
                    correct_options_set = set(correct_options)
                    user_options_set = set(user_options)

                    if user_options_set == correct_options_set:
                        correct_count += 1
                        is_correct = True
            else:
                # 其他题型的处理
                if isinstance(answer, str):
                    user_option = answer.split(":")[0].strip().upper()
                    if user_option in correct_options:
                        correct_count += 1
                        is_correct = True

            # 生成题目结果详情
            color = "green" if is_correct else "red"
            result_details.append(f'<div style="color: {color};">')
            result_details.append(f"问题 {idx+1}：{question['question']}")
            result_details.append(f"你的答案：{answer}")
            result_details.append(f"正确答案：{question['correct']}")
            if not is_correct:
                result_details.append(f"解析：{question['explanation']}")
                # 更新错题集
                if question_type == "多项选择题":
                    last_mcq_index += 1
                    mistake_book["mcq"][str(last_mcq_index)] = question
                else:
                    last_other_index += 1
                    mistake_book["other"][str(last_other_index)] = question
            result_details.append('</div>\n')

        # 更新现有的错题集
        MISTAKE_JSON["mcq"].update(mistake_book["mcq"])
        MISTAKE_JSON["other"].update(mistake_book["other"])

        if study_or_review == "study":

            # 将更新后的错题集写回文件
            with open(os.path.join(mistake_book_dir, f"Mistake_Book_{User_id}.json"), 'w', encoding='utf-8') as M_file:
                json.dump(MISTAKE_JSON, M_file, ensure_ascii=False, indent=4)

        if len(quiz)==0:
            return gr.update(value="当前没有题目，不可提交", visible=True)
        # 生成结果报告
        result = f"<h3>你的得分是: {correct_count} / {len(quiz)}</h3>\n\n"
        result += "\n".join(result_details)

        if correct_count / len(quiz) !=1 and study_or_review == "study":
            result += "<p>已加入错题集。</p>"

        return gr.update(value=result, visible=True)

    def refresh_mistake_tab(self, question_type,User_id):
        forgotten_questions = self.get_random_questions(User_id)
        if question_type == "多项选择题":
            return self.question_show_tab(question_type, forgotten_questions[0])
        return [gr.update(visible=False)]+self.question_show_tab(question_type, forgotten_questions[1])

    def get_random_questions(self,User_id, num_mcq=5, num_other=5):
        try:
            with open(os.path.join(mistake_book_dir, f"Mistake_Book_{User_id}.json"), 'r', encoding='utf-8') as M_file:
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
        self.mistake_book_quiz = [selected_mcq, selected_other]

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


    def get_response(self, history_list,User_id):
        if self.count % 20==0:
            self.information=self.get_random_questions(User_id,10,10)
            self.information=json.dumps(self.information[0], ensure_ascii=False) + " " +json.dumps(self.information[1], ensure_ascii=False)
        for response_message in generate_response(history_list, self.information, User_id):
            yield response_message

    def add_msg(self,user_message, history):
        return "", history + [[user_message, None]]