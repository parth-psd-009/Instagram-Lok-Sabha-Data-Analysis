import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from pathlib import Path

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    """
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' is None.")
    image = cv2.imread(image_path)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 150% of its original size for better recognition of small text
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    # Sharpen the image to enhance text clarity
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    # Apply adaptive thresholding to the sharpened image
    thresholded = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresholded

def extract_text_from_image(image_path, lang_code):
    """
    Extract text from an image using Tesseract OCR with the specified language.
    """
    if image_path is None or len(image_path) == 0:
        raise ValueError("Image path is None or empty.")
    if lang_code is None:
        raise ValueError("Language code is None.")

    processed_image = preprocess_image(image_path[0])  # Access the first element of the list
    text = pytesseract.image_to_string(processed_image, lang=lang_code)
    print(f"TEXT: {text}")
    return text

def process_images_from_df(df): 
    """
    Process images from a DataFrame, perform OCR with the specified language, and display the results.
    """
    extracted_texts = []

    for idx, row in df.iterrows():
        image_paths = row['Image_path']
        lang_code = row['languageCode']

        try:
            if not image_paths or image_paths[0] is None:
                raise ValueError("Image path is None or empty.")

            print(f"Processing file: {image_paths[0]} with language code: {lang_code}")

            # Extract text from the image using the specified language
            text = extract_text_from_image(image_paths, lang_code)
            extracted_texts.append(text)
            print(f"Extracted Text:\n{text}")

            # Display the processed image
            processed_image = preprocess_image(image_paths[0])
            cv2.imshow('Processed Image', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print('\n----------\n')
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing file '{image_paths}': {e}")
            extracted_texts.append(None)
        except Exception as e:
            print(f"Unexpected error processing file '{image_paths}': {e}")
            extracted_texts.append(None)

    df['extracted_text'] = extracted_texts

if __name__ == "__main__":
    # Example DataFrame
    input_path = "../json files/Loksabha_t_images.json"
    output_path = "../json files/"
    df = pd.read_json(input_path)

    process_images_from_df(df)
