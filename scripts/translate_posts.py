import json
import os
import pandas as pd
from googletrans import Translator

translator = Translator()

def g_trans(text_string):
    try:
        translated = translator.translate(text_string, dest='en')
        if translated is not None:
            response = {
                'src': translated.src,
                'dest': translated.dest,
                'text': translated.text,
                'pronunciation': translated.pronunciation
            }
            return response
        else:
            print("Failed to translate the text.")
            return {'text': ""}
    except Exception as e:
        print("An error occurred during translation:", str(e))
        return {'text': ""}

def translate_file(directory, filename, column_name):
    file_path = os.path.join(directory, filename)
    df = pd.read_json(file_path)
    
    if f'translated_{column_name}' not in df.columns:
        df[f'translated_{column_name}'] = pd.NA
    
    for i in range(len(df)):
        if column_name in df.columns and pd.isna(df.at[i, f'translated_{column_name}']) and not pd.isna(df.at[i, column_name]):
            text = df.at[i, column_name]
            response = g_trans(text)
            df.at[i, f'translated_{column_name}'] = response['text']
            print(f"Translated {column_name} for index {i}")
    
    output_path = os.path.join(directory, "Loksabha_t.json")
    df.to_json(output_path, orient='records', indent=4)

if __name__ == "__main__":
    column_name = input("Enter the name of the column to translate: ")
    translate_file("../json files", "Loksabha.json", column_name)
