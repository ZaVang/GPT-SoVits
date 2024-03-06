from abc import abstractmethod
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json


class ModelHandler:
    def __init__(self, models_folder, models_info_file='models_info.json'):
        self.models_folder = models_folder
        self.models_info_file = models_info_file
        self.models_info = {}
        self.load_models_info()


    def load_models_info(self):
        """加载模型信息JSON文件"""
        self.update_models_info()
        with open(os.path.join(self.models_folder, self.models_info_file), 'r') as file:
            self.models_info = json.load(file)


    def update_models_info(self):
        """更新模型信息，并写入JSON文件"""
        for character in os.listdir(self.models_folder):
            character_path = os.path.join(self.models_folder, character)
            if os.path.isdir(character_path):
                self.models_info[character] = {}
                for model_file in os.listdir(character_path):
                    model_path = os.path.join(character_path, model_file)
                    if os.path.isfile(model_path):
                        self.models_info[character][model_file] = model_path
        
        with open(os.path.join(self.models_folder, self.models_info_file), 'w') as file:
            json.dump(self.models_info, file, indent=4)

