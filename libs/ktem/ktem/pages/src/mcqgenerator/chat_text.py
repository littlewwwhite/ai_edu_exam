from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
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
        从根源来讲，深度学习是机器学习的一个分支，是指一类问题以及解决这类问题的方法。
        首先，深度学习问题是一个机器学习问题，指从有限样例中通过算法总结出一般性的规律，并可以应用到新的未知数据上。
        比如，我们可以从一些历史病例的集合中总结出症状和疾病之间的规律。
        这样当有新的病人时，我们可以利用总结出来的规律，来判断这个病人得了什么疾病。
        其次，深度学习采用的模型一般比较复杂，指样本的原始输入到输出目标之间的数据流经过多个线性或非线性的组件（component）。
        因为每个组件都会对信息进行加工，并进而影响后续的组件，所以当我们最后得到输出结果时，我们并不清楚其中每个组件的贡献是多少。
        这个问题叫作贡献度分配问题（CreditAssignment Problem，CAP）[Minsky, 1961]。
        """

        # Initialize external memory store for session history
        self.store = {}

        sys_pro = """
        # Role: 您是一位善于与学生交流的虚拟AI老师。

        ## Goals:
        1. 您需要根据学生之间的错题内容给学生问问题（简答题），考验他对错题的知识点是否已经掌握
        2. 当学生回答后，如果答案正确，请确认学生答对了，并继续出下一题。
        3. 如果答案错误，请分析学生的回答，指出错误所在，结合题目和已有信息引导学生重新思考并回答。
        4. 学生有两次答题机会，如果第二次回答仍然错误，请给出正确答案并详细解释学生的错误点。
        5. 通过引导性的提示，帮助学生逐步提高对知识点的理解。
        6. 根据学生的回答质量，分析学生的学习能力，强项，弱项。来及时调整对话和题目的设计。


        ## Attention:
        - 优秀的开放式题目是我们评估学生对知识掌握程度的重要工具。如果题目设计不当，无法准确反映学生的实际水平，可能会导致筛选出的学生并不真正掌握学科知识。、
        - 对话中应该体现出提升学生的批判性思维与创造力，引导学生深入理解知识，给予适当程度的鼓励和批评
        - 交互模式可自由发挥，但必须遵守Goals


        ## Text:
        你需要根据以下文本来设计合理的题目，以下是知识点文本：
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

    def generate_response(self, user_input, session_id="1"):
        # 获取当前会话的历史记录
        history = self.get_session_history(session_id)



        # 生成响应
        response_message = self.chain_with_message.invoke(
            {"question":user_input,"info": self.information},
            config={"configurable": {"session_id": session_id}}
        )



        # 初始化空的 chunks 变量来累积响应
        chunks = ""

        # 提取响应内容
        content = response_message.content  # 直接访问内容
        for chunk in content:
            chunks += chunk
            # 返回完整的对话历史和当前累积的响应
            chat_history = [(message.content, None) if isinstance(message, HumanMessage) else (None, message.content)
                            for message in history.messages]
            yield chat_history, ""

    def ui(self):
        with gr.Blocks(css=custom_css) as self.mcq_interface:
            gr.Markdown("# MCQ Generator")

            with gr.Tab("对话错题", visible=True) as chat_tab:
                self.chatbot = gr.Chatbot(
                    placeholder="根据错题集对话\n请你回答问题",
                    show_label=False,
                    elem_id="main-chat-bot",
                    show_copy_button=True,
                    likeable=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    self.text_input = gr.Textbox(
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
                with gr.Row():

                    pass

                # Bind input submit and button click to generate response
                self.text_input.submit(
                    fn=self.generate_response,
                    inputs=[self.text_input, gr.State("1")],
                    outputs=[self.chatbot, self.text_input]
                )
                self.submit_btn.click(
                    fn=self.generate_response,
                    inputs=[self.text_input, gr.State("1")],
                    outputs=[self.chatbot, self.text_input]
                )


mcq_generator = GradioMCQPage()
mcq_generator.ui()
mcq_generator.mcq_interface.launch(
    server_name="172.18.232.73",
    server_port=7864,
    share=True,
    inbrowser=False,
)
