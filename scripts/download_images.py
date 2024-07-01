import json
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

class ImageExtractor:
    def __init__(self, json_file_path, save_directory):
        self.json_file_path = json_file_path
        self.save_directory = save_directory
        self.df = None

    def load_json_to_df(self):
        try:
            self.df = pd.read_json(self.json_file_path)
        except FileNotFoundError:
            print(f"Error: The file {self.json_file_path} was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON file.")
        except Exception as e:
            print(f"An error occurred while loading the JSON file: {e}")

    def save_images_from_urls(self):
        if self.df is None:
            print("Error: DataFrame is not initialized. Load the JSON file first.")
            return

        if not os.path.exists(self.save_directory):
            try:
                os.makedirs(self.save_directory)
            except OSError as e:
                print(f"Error: Failed to create directory {self.save_directory}. {e}")
                return

        image_paths = []

        for i, row in self.df.iterrows():
            media_list = row.get('media', [])
            if not isinstance(media_list, list):
                print(f"Warning: Media at row {i} is not a list. Skipping.")
                image_paths.append([])
                continue

            paths = []
            for media in media_list:
                if media.get('type') == 'photo':
                    img_url = media.get('url')
                    if img_url:
                        try:
                            response = requests.get(img_url)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            img_name = f"image_{i}_{len(paths)}.jpg"
                            img_path = os.path.join(self.save_directory, img_name)
                            image.save(img_path)
                            print(f"Image saved: {img_name}")
                            paths.append(img_path)
                        except requests.RequestException as req_err:
                            print(f"Error downloading image from {img_url}: {req_err}")
                            paths.append(None)
                        except IOError as io_err:
                            print(f"Error saving image: {io_err}")
                            paths.append(None)
            image_paths.append(paths)

        self.df['Image_path'] = image_paths

    def process(self):
        self.load_json_to_df()
        self.save_images_from_urls()
        if self.df is not None:
            output_path = os.path.join(self.save_directory, "output_with_image_paths.json")
            try:
                self.df.to_json(output_path, orient='records', indent=4)
                print(f"Processed data saved to {output_path}")
            except Exception as e:
                print(f"Error saving the DataFrame to JSON: {e}")

if __name__ == "__main__":
    json_file_path = "D:\\instagramproject\\json files\\Loksabha_t_descCluster.json"  # Update this path
    save_directory = "../images"  # Update this path

    extractor = ImageExtractor(json_file_path, save_directory)
    extractor.process()
