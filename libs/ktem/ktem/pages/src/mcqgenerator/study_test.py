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

            # 设置刷新按钮的回调
            self.refresh_btn.click(
                fn=self.refresh_mistake_tab,
                inputs=[self.question_type],
                outputs=[item for sublist in self.mistake_question_block for item in sublist]  # 展开所有问题和选择的组件
            )

            self.generate_btn.click(
                fn=self.generate_mcqs,
                inputs=[self.file_upload, self.text_input, self.mcq_count, self.subject, self.tone, self.question_type],
                outputs=[item for sublist in self.question_block for item in sublist]  # 展开所有问题和选择的组件
            )

            self.start_study_btn.click(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=study_tab
            )

            self.submit_btn.click(
                fn=self.evaluate_answers,
                inputs=[self.question_type]+[opt[1] for opt in self.question_block],
                outputs=self.score_display
            )

    def create_question_block(self, idx):
        question_text = gr.Markdown(visible=False)
        radio_buttons = gr.Radio(choices=[], visible=False)
        checkbox_buttons = gr.CheckboxGroup(choices=[], visible=False)
        return question_text, radio_buttons, checkbox_buttons

    def generate_mcqs(self, file, text, mcq_count, subject, tone, question_type):
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
        # self.question_block.clear()
        # # 必须得用传进来的参数question_type，其余方式不行
        # self.question_block = [self.create_question_block(i, question_type) for i in range(20)]

        question_updates = []

        for idx, data in enumerate(self.quiz_data.values()):
            question_display = f"### 问题 {idx + 1}: {data['question']}"
            choices = [f"{key}: {value}" for key, value in data['options'].items()]
            if question_type == "多项选择题":
                # 显示 CheckboxGroup 组件，隐藏 Radio 组件
                question_updates.extend([
                    gr.update(value=question_display, visible=True),
                    gr.update(visible=False),  # Radio 组件
                    gr.update(choices=choices, visible=True, interactive=True)  # CheckboxGroup 组件
                ])
            else :
                # 显示 Radio 组件，隐藏 CheckboxGroup 组件
                question_updates.extend([
                    gr.update(value=question_display, visible=True),
                    gr.update(choices=choices, visible=True, interactive=True),  # Radio 组件
                    gr.update(visible=False)  # CheckboxGroup 组件
                ])

            # 隐藏多余的问题块
        for _ in range(len(self.quiz_data), len(self.question_block) * 3):
            question_updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
        return question_updates


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
            return f"评分规则：全选得满分，少选、多选、错选均不得分\n你的得分是: {correct_count} / {len(self.quiz_data)}"
        else:
            return f"评分规则：选对得分\n你的得分是: {correct_count} / {len(self.quiz_data)}"

    def refresh_mistake_tab(self, question_type):
        forgotten_questions = self.get_random_questions()
        if question_type == "多项选择题":
            return self.update_mistake_tab(question_type, forgotten_questions[0])
        return self.update_mistake_tab(question_type, forgotten_questions[1])


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



    def update_mistake_tab(self, question_type, forgotten_questions):
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
    server_port=7862,
    share=True,
    inbrowser=False,
)
