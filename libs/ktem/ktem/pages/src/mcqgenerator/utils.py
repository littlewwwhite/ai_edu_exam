import os
import json
import re

import PyPDF2
import pandas as pd
import traceback


def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader=PyPDF2.PdfReader(file)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            return text
        except Exception as e:
            raise Exception("error reading the PDF file")
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    elif file.name.endswith(".rtf"):
        return file.read().decode("utf-8")
    
    elif file.name.endswith(".docx"):
        return file.read()
    
    elif file.name.endswith(".doc"):
        return file.read()
    
    #elif file.name.endswith(".jpeg"):
        #return read_text_from_jpeg_image(file)
    
    #elif file.name.endswith(".png"):
        #image = Image.open(file)  # Replace 'image.jpg' with the path to your image file
        # Use pytesseract to do OCR on the image
        #return pytesseract.image_to_string(image)
         

    
    else:
        raise Exception("unsopported file formate only pdf and text file supported")
    
def get_table_data(quiz_str):
    try:
        #convert the quiz from string to dict
        print(quiz_str)
        quiz_str_list=re.findall(r'```json([^`]+?)```', quiz_str)
        print(quiz_str_list)
        if quiz_str_list:
            quiz_str=quiz_str_list[0]
        quiz_dict=json.loads(quiz_str)
        # quiz_table_data=[]

        # for key,value in quiz_dict.items():
        #     mcq=value["mcq"]
        #     options= " || ".join([
        #         f"{option}->{option_value}" for option,option_value in value["options"].items()
        #     ])
        #
        #     correct=value["correct"]
        #     quiz_table_data.append({"MCQ":mcq,"Choices":options,"Correct":correct})
        return quiz_dict
    except Exception as e:
        traceback.print_exception(type(e),e,e.__traceback__)
        return False
    

#def read_text_from_jpeg_image(image_file):
    # Create a BytesIO object containing the JPEG image data
    #print(image_file)
    #image_data = image_file.read()
    #image_bytes_io = io.BytesIO(image_data)
    
    # Open the image BytesIO object using PIL
    #image = Image.open(image_bytes_io)
    
    # Use pytesseract to do OCR on the image
    #text = pytesseract.image_to_string(image)
    
    #return text