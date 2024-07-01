import pandas as pd
import ollama
import os
import time

# Define the function to generate image captions
def generate_caption(image_path):
    if not image_path or image_path == '[None]':
        return None
    
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        stream = ollama.generate(
            model='llava:34b',
            # model = 'llava:v1.6',
            prompt='describe this image and make sure to include anything notable about it (include text you see in the image) and any political, social and cultural references:',
            images=[image_bytes],
            stream=True
        )
        
        response = ""
        for chunk in stream:
            response += chunk['response']
        # print(f"For image path {image_path}, the response is:\n{response}")
        return response
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

input_df = pd.read_json('../json files/Loksabha_t_images.json')
input_df['Image_path'] = input_df['Image_path'].apply(lambda x: os.path.abspath(x[0]) if x and x[0] else None)
start_index = int(input("Enter the starting index for processing: ")) #takes input from the user which is the starting index of the dataframe
output_path = '../json files/Loksabha_t_images_itt.json'
if os.path.exists(output_path):
    output_df = pd.read_json(output_path, orient='split')
else:
    output_df = input_df.copy()
    output_df['Image_caption'] = None
print('Starting inferences')
for idx in range(start_index, len(input_df)):
    image_path = input_df.at[idx, 'Image_path']
    if pd.notna(output_df.at[idx, 'Image_caption']):
        print(f"Skipping already processed image {image_path}")
        continue
    caption = generate_caption(image_path)
    output_df.at[idx, 'Image_caption'] = caption
    output_df.to_json(output_path, orient='split', index=False)
    print(f"Saved output to {output_path} for instance {idx} after processing image {image_path}")
    time.sleep(5)

print("Finished inferencing")
print(f"Final output saved to {output_path}")
