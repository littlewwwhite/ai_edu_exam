from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.docstore.document import Document



import gradio as gr
from langchain_openai import ChatOpenAI

custom_css = """
#main-chat-bot {
    border: none !important; /* 移除外层边框 */
}
"""

class GradioMCQPage:
    def __init__(self):
        self.quiz_data = {}
        self.user_answers = {}
        self.TYPE = ["多项选择题", "单项选择题", "对错题", "填空题"]
        self.information = """
         "1": {
            "question": "下列哪个选项最准确地描述了前馈神经网络的特点？",
            "options": {
                "A": "从前隐藏层到输出层的路径中允许存在循环连接。",
                "B": "通过时间递归处理序列数据，适合自然语言处理任务。",
                "C": "信息仅从输入层单向传递至输出层，不包含任何反馈连接。",
                "D": "主要用于图像识别任务，具备局部感受野和参数共享机制。"
            },
            "correct": "C",
            "explanation": "前馈神经网络的信息只沿着一个方向从前隐藏层到输出层进行传递，没有反馈回路或循环结构。选项A描述了具有循环连接的神经网络（如RNN），B则是对循环神经网络特点的描述，而D则适用于卷积神经网络的特点。"
        },
        "2": {
            "question": "关于反向传播算法在训练神经网络中的作用，以下哪个说法是正确的？",
            "options": {
                "A": "该算法通过随机选取权重来加速神经网络的学习过程。",
                "B": "它负责优化模型的超参数，如学习率和批量大小。",
                "C": "反向传播是一种用于更新神经网络中的权重的方法，以最小化损失函数。",
                "D": "此方法主要用于数据预处理阶段，增强输入特征的质量。"
            },
            "correct": "C",
            "explanation": "反向传播算法的核心作用是通过计算损失关于每个参数的梯度来更新神经网络的权重，目标是最小化预测输出与实际标签之间的差距（即损失函数）。选项A、B和D分别描述了不准确或无关的做法。"
        },
        "3": {
            "question": "卷积神经网络广泛应用于图像识别任务的主要原因是什么？",
            "options": {
                "A": "因为它们能够捕捉到序列数据的时间依赖关系。",
                "B": "可以实现局部感受野和参数共享，提高对平移不变性的处理能力。",
                "C": "通过增加隐藏层的深度来降低模型复杂度并加速训练速度。",
                "D": "卷积神经网络可以通过调整学习率自动适应不同的任务需求。"
            },
            "correct": "B",
            "explanation": "卷积神经网络（CNN）特别适合图像识别，主要得益于其局部感受野和参数共享的特性，这有助于模型捕捉到输入信号中空间结构信息，并且保持对平移变换的不变性。选项A描述了RNN的特点；C、D则不准确地反映了训练或架构设计的原则。"
        },
        "4": {
            "question": "注意力机制允许模型在处理序列数据时只关注输入中的关键部分。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "A",
            "explanation": "注意力机制通过赋予不同位置不同的权重，使得模型能够更加聚焦于重要的信息。"
        },
        "5": {
            "question": "自注意力机制只允许序列中的每个元素关注其前一个元素。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "B",
            "explanation": "自注意力机制让序列中每一个位置都可以从其他所有位置获取信息，而不只是前一个。这一特性使得模型能够学习到更复杂的依赖关系。"
        },
        "6": {
            "question": "计算查询向量和键值对之间的相似性是得出注意力权重的过程。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "A",
            "explanation": "在注意力机制中，通过点积或加权求和等方式比较查询与键的相似性来确定注意力权重。"
        },
        "7": {
            "question": "自注意力机制只允许序列中的每个元素关注其前一个元素。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "B",
            "explanation": "自注意力机制让序列中每一个位置都可以从其他所有位置获取信息，而不只是前一个。这一特性使得模型能够学习到更复杂的依赖关系。"
        },
        "8": {
            "question": "计算查询向量和键值对之间的相似性是得出注意力权重的过程。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "A",
            "explanation": "在注意力机制中，通过点积或加权求和等方式比较查询与键的相似性来确定注意力权重。"
        },
        "9": {
            "question": "科学研究成果应当在公布前严格保密。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "A",
            "explanation": "科学研究中的许多创新和发现，在正式发表之前需要保持严格的保密状态，以防止机密信息泄露或被竞争对手非法利用。"
        },
        "10": {
            "question": "科学保密仅适用于军事领域。",
            "options": {
                "A": "True",
                "B": "False"
            },
            "correct": "B",
            "explanation": "虽然军事领域的科学研究经常需要高度保密，但许多其他类型的科研工作（如生物技术、信息技术等）同样可能包含敏感信息，也需要适当的保密措施。"
        },
        """

        # Initialize external memory store for session history
        self.store = {}

        sys_pro = """
        # Role: 您是一位善于与学生交流的虚拟AI老师。

        ## Goals:
        - 您需要整合学生之前的错题内容，并给学生出题（简答题），考验他对错题的知识点是否已经掌握
        - 如果答案错误，请分析学生的回答，指出错误所在，结合题目和已有信息引导学生重新思考并回答。        
        - 根据学生的回答质量，分析学生的学习能力，强项，弱项。来及时调整对话和题目的设计。


        ## Attention:
        - 优秀的开放式题目是我们评估学生对知识掌握程度的重要工具。如果题目设计不当，无法准确反映学生的实际水平，可能会导致筛选出的学生并不真正掌握学科知识。、
        - 对话中应该体现出提升学生的批判性思维与创造力，引导学生深入理解知识，给予适当程度的鼓励和批评
        - 将原错题整理为简答题，不要以原题的形式出题
        - 需要将错题集之间相关的知识点进行整合并出题
        - 当学生回答后，如果答案正确，请确认学生答对了，并继续出下一题。
        - 学生有两次答题机会，如果第二次回答仍然错误，请给出正确答案并详细解释学生的错误点。
        - 当这一道题目结束后，紧接着给学生出第二道题，不要等学生主动要求出题

        ## claim
        - 每次仅提出一道问题考察学生
        - 在提问中不要给出答案
        - 学生回答完后要给出解析，再给出下一道题

         

        ## Text:
        以下是学生的错题集：
        {info}
        """
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages([
            ("system",sys_pro),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])


        # Set up the language model and memory chain
        chain = prompt | ChatOpenAI(api_key="sk-c9ee10eb8856450cab089c787622526a",
                              base_url="https://api.deepseek.com",
                              model="deepseek-chat",
                              top_p=0.7,
                              temperature=0.95)
        self.chain_with_message = RunnableWithMessageHistory(chain, self.get_session_history,input_messages_key="question",
                history_messages_key="history",)

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def add_msg(self,user_message, history):
        return "", history + [[user_message, None]]

    def generate_response(self,history_list, session_id="1"):
        # 获取当前会话的历史记录
        print(history_list)
        history = self.get_session_history(session_id)

        # 返回完整的对话历史
        chat_history = [[message.content, None] if isinstance(message, HumanMessage) else [None, message.content]
                        for message in history.messages]
        # 初始化空的 chunks 变量来累积响应
        chunks = ""
        history_list[-1][1] = ""
        # 提取响应内容
        for chunk in self.chain_with_message.stream({"question":history_list[-1][0],"info": self.information},
                                                    config={"configurable": {"session_id": session_id}}):
            history_list[-1][1] += chunk.content
            yield history_list

    # for chunk in chain_with_message.stream({"question": user_input, "info": information},
    #                                        config={"configurable": {"session_id": 1}}):
    #     print(chunk.content, end='', flush=True)

    def ui(self):
        with gr.Blocks(css=custom_css) as self.mcq_interface:
            gr.Markdown("# MCQ Generator")

            with gr.Tab("对话错题", visible=True) as chat_tab:
                self.chatbot = gr.Chatbot(
                    placeholder="根据错题集对话\n请你回答问题,按下回车开始。",
                    show_label=False,
                    elem_id="main-chat-bot",
                    show_copy_button=True,
                    likeable=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    self.text_input = gr.Textbox(
                        placeholder="输入你的答案",
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


                # Bind input submit and button click to generate response
                self.text_input.submit(
                    self.add_msg,
                    [self.text_input, self.chatbot],
                    [self.text_input, self.chatbot],
                    queue=False).then(
                    fn=self.generate_response,
                    inputs=[self.chatbot, gr.State("1")],
                    outputs=[self.chatbot]
                )
                self.submit_btn.click(
                    self.add_msg,
                    [self.text_input, self.chatbot],
                    [self.text_input, self.chatbot],
                    queue=False).then(
                    fn=self.generate_response,
                    inputs=[self.chatbot, gr.State("1")],
                    outputs=[self.chatbot]
                )



# mcq_generator = GradioMCQPage()
# mcq_generator.ui()
# mcq_generator.mcq_interface.launch(
#     server_name="172.18.232.73",
#     server_port=7864,
#     share=True,
#     inbrowser=False,
# )


