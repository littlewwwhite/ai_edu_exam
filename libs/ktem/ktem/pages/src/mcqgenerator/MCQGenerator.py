import os
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from ktem.pages.src.mcqgenerator.utils import read_file,get_table_data
from ktem.pages.src.mcqgenerator.logger import logging
from ktem.pages.src.mcqgenerator.llm import get_llm

#importing the necessary packages from langchain

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from ktem.pages.src.mcqgenerator.prompt import template, template2

# load_dotenv()  # take environment variables from .env.
# KEY=os.getenv("my-openkey")
# llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo", temperature=0.7)
# llm=ChatOpenAI(openai_api_key="67f836edc96405d0c4eea5d8eeff70d0.pVev8zx1tG17ohTQ",model_name="glm-4", temperature=0.7, base_url="https://open.bigmodel.cn/api/paas/v4/")
llm = ChatOpenAI(openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                 base_url=os.environ.get('DEEPSEEK_API_URL'),
                 model="deepseek-chat",
                 top_p=0.7,
                 temperature=0.5)
# model=os.environ.get('default_model')
# llm, _ = get_llm(model)
# template="""
# Text:{text}
# You are an expert MCQ maker. Given the above text, it is your job to \
# create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone.
# Make sure the questions are not repeated and check all the questions to be conforming the text as well.
# Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
# Ensure to make {number} MCQs
# {response_json}
#
# """

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json", "examples", "type_", "format_"],
    template=template
    )

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

# template2="""
# You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
# You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
# if the quiz is not at per with the cognitive and analytical abilities of the students,\
# update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
# Quiz_MCQs:
# {quiz}
#
# Check from an expert English Writer of the above quiz:
# """

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz", "type_", "tone", "format_"], template=template2)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json", "examples", "type_", "format_"],
                                        output_variables=["quiz", "review"], verbose=True)






