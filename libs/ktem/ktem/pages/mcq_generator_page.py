import gradio as gr
from langchain_community.callbacks import get_openai_callback
from typing import List, Dict
import json
import asyncio
import sys

sys.path.append("src")
from ktem.pages.src.mcqgenerator.utils import read_file
from ktem.pages.src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from ktem.pages.src.mcqgenerator.RAG import RAG
from ktem.pages.src.mcqgenerator.constants import type_of_question_info

class GradioMCQPage:
    def __init__(self, app):
        self._app = app
        self.quiz_data = {}
        self.edit_mode = False

    def ui(self):
        with gr.Blocks() as self.mcq_interface:
            gr.Markdown("# MCQ Generator")
            
            with gr.Row():
                with gr.Column(scale=1):
                    self.file_upload = gr.File(label="Upload a PDF, RTF, or TXT file")
                    self.text_input = gr.Textbox(label="Or enter your text here", lines=5)
                    self.mcq_count = gr.Slider(minimum=3, maximum=50, value=5, step=1, label="Number of MCQs")
                    self.subject = gr.Textbox(label="Subject")
                    self.tone = gr.Textbox(label="Complexity Level of Questions", placeholder="Simple")
                    self.question_type = gr.Dropdown(choices=["多项选择题", "填空题"], label="Type of question")
                    self.rag_mode = gr.Dropdown(choices=["mode1", "mode2"], label="RAG mode")
                    self.llm_model = gr.Dropdown(choices=["model1", "model2"], label="LLM model")
                    self.graph_type = gr.Dropdown(choices=["graph1", "graph2"], label="Graph type")
                    self.generate_btn = gr.Button("Generate MCQs")

                with gr.Column(scale=2):
                    self.output_area = gr.Markdown("Generated MCQs will appear here")
                    self.edit_mode_checkbox = gr.Checkbox(label="Edit Mode", value=False)
                    self.save_btn = gr.Button("Save Changes")
                    self.export_btn = gr.Button("Export Quiz")

            self.generate_btn.click(fn=self.generate_mcqs, 
                                    inputs=[self.file_upload, self.text_input, self.mcq_count, self.subject, 
                                            self.tone, self.question_type, self.rag_mode, self.llm_model, 
                                            self.graph_type],
                                    outputs=[self.output_area])
            
            self.edit_mode_checkbox.change(fn=self.toggle_edit_mode, outputs=[self.output_area])
            self.save_btn.click(fn=self.save_changes, outputs=[self.output_area])
            self.export_btn.click(fn=self.export_quiz, outputs=[gr.File()])

    async def generate_mcqs(self, file, text, mcq_count, subject, tone, question_type, rag_mode, llm_model, graph_type):
        try:
            if file:
                text = read_file(file)
            elif not text:
                return "Please upload a file or enter some text."

            result = await RAG(question=text, mode=rag_mode, model=llm_model, graph_type=graph_type)
            
            if result["message"] == "I couldn't find any relevant documents to answer your question." or result["message"] == "Something went wrong":
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

            self.quiz_data = response.get("quiz", {})
            return self.format_quiz_for_display(self.quiz_data)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def format_quiz_for_display(self, quiz_data: Dict) -> str:
        formatted_output = ""
        for idx, (_, question) in enumerate(quiz_data.items(), start=1):
            formatted_output += f"### Question {idx}: {question['question']}\n\n"
            for opt, text in question['options'].items():
                formatted_output += f"- {opt}: {text}\n"
            formatted_output += f"\nCorrect Answer: {question['correct']}\n"
            formatted_output += f"Explanation: {question['explanation']}\n\n"
        return formatted_output

    def toggle_edit_mode(self, edit_mode):
        self.edit_mode = edit_mode
        return self.format_quiz_for_edit() if edit_mode else self.format_quiz_for_display(self.quiz_data)

    def format_quiz_for_edit(self) -> str:
        formatted_output = ""
        for idx, (_, question) in enumerate(self.quiz_data.items(), start=1):
            formatted_output += f"### Question {idx}:\n"
            formatted_output += f"Question: {question['question']}\n"
            for opt, text in question['options'].items():
                formatted_output += f"Option {opt}: {text}\n"
            formatted_output += f"Correct Answer: {question['correct']}\n"
            formatted_output += f"Explanation: {question['explanation']}\n\n"
        return formatted_output

    def save_changes(self):
        # In a real implementation, you would parse the edited text and update self.quiz_data
        return "Changes saved successfully!"

    def export_quiz(self):
        return gr.File.update(value=json.dumps(self.quiz_data, ensure_ascii=False, indent=2), 
                              visible=True, 
                              label="Download Quiz JSON")