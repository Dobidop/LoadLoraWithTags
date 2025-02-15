import os
import folder_paths
import hashlib
import requests
import json
import re

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
ANY = AnyType("*")

def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_dict_to_json(data_dict, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
    except Exception as e:
        print(f"Error saving JSON to file: {e}")

def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)
    return response.json() if response.status_code == 200 else None

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def extract_lora_references(prompt_text):
    pattern = r"<lora:([^>]+)>"
    return re.findall(pattern, prompt_text)

class FetchLoraTags:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": {
                "selection_mode": (["From Selected Lora", "From Text Prompt", "From Lora Stack"], {"default": "From Selected Lora"},),
                "lora_name": (LORA_LIST,),
                "print_tags": ("BOOLEAN", {"default": False}),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "use_header_format": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "prompt_text": ("STRING", {
                "default": "",
                "multiline": False,
                "defaultInput": True,
            }),
        },
    }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "fetch_trigger_tags"
    CATEGORY = "loaders"

    def fetch_trigger_tags(self, selection_mode, lora_name, print_tags, force_fetch, use_header_format, prompt_text = "", lora_stack = ""):
        json_tags_path = "./loras_tags.json"
        lora_tags = load_json_from_file(json_tags_path)

        if selection_mode == "From Text Prompt":
            selected_loras = extract_lora_references(prompt_text)
        elif selection_mode == "From Lora Stack":
            selected_loras = [entry[0] for entry in lora_stack]
        else:
            selected_loras = [lora_name]

        formatted_output = []

        for lora in selected_loras:
            if not lora.endswith(".safetensors"):
                lora += ".safetensors"

            lora_path = folder_paths.get_full_path("loras", lora)

            if lora_path is None or not os.path.exists(lora_path):
                print(f"Warning: LoRA file '{lora}' was not found in the designated folder paths.")
                continue

            output_tags = lora_tags.get(lora, None) if lora_tags is not None else None

            if output_tags is None or force_fetch:
                print(f"Calculating hash for {lora_path}")
                LORAsha256 = calculate_sha256(lora_path)
                model_info = get_model_version_info(LORAsha256)
                if model_info and "trainedWords" in model_info:
                    output_tags = model_info["trainedWords"]
                    lora_tags[lora] = output_tags
                    save_dict_to_json(lora_tags, json_tags_path)
                else:
                    output_tags = []
                    lora_tags[lora] = output_tags
                    save_dict_to_json(lora_tags, json_tags_path)

            if output_tags:
                if use_header_format:
                    formatted_output.append(f"{lora}:")
                    formatted_output.extend(output_tags)
                else:
                    formatted_output.append(", ".join(output_tags))
                
                if print_tags:
                    print(f"Tags for {lora}: {', '.join(output_tags)}")

        return ("\n".join(formatted_output),)


class LoraSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        LORA_LIST = ["None"] + sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": {
                "lora_name_1": (LORA_LIST,),
            },
            "optional": {
                "lora_name_2": (LORA_LIST,),
                "lora_name_3": (LORA_LIST,),
                "lora_name_4": (LORA_LIST,),
                "lora_name_5": (LORA_LIST,),
            }
        }
    
    RETURN_TYPES = (ANY, ANY, ANY, ANY, ANY, "LORA_STACK",)
    RETURN_NAMES = ("lora_name_1", "lora_name_2", "lora_name_3", "lora_name_4", "lora_name_5", "lora_stack")
    OUTPUT_IS_LIST = (False, False, False, False, False, False,)
    
    FUNCTION = "returnLoraSelections"
    CATEGORY = "loaders"

    def returnLoraSelections(self, lora_name_1="None", lora_name_2="None", lora_name_3="None", lora_name_4="None", lora_name_5="None"):
        # List of selected Loras (excluding "None")
        selected_loras = [lora for lora in (lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5) if lora != "None"]

        # Convert into lora stack format (default strengths: 1.0, 0.0)
        lora_stack = [(lora, 1.0, 0.0) for lora in selected_loras]

        return lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5, lora_stack



class LoraLoaderTagsQuery:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (LORA_LIST, ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                              "query_tags": ("BOOLEAN", {"default": True}),
                              "tags_out": ("BOOLEAN", {"default": True}),
                              "print_tags": ("BOOLEAN", {"default": False}),
                              "bypass": ("BOOLEAN", {"default": False}),
                              "force_fetch": ("BOOLEAN", {"default": False}),
                              },
                "optional":
                            {
                                "opt_prompt": ("STRING", {"forceInput": True}),
                            }
                }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, query_tags, tags_out, print_tags, bypass, force_fetch, opt_prompt=None):
        if strength_model == 0 and strength_clip == 0 or bypass:
            if opt_prompt is not None:
                out_string = opt_prompt
            else:
                out_string = ""
            return (model, clip, out_string,)
        
        json_tags_path = "./loras_tags.json"
        lora_tags = load_json_from_file(json_tags_path)
        output_tags = lora_tags.get(lora_name, None) if lora_tags is not None else None
        if output_tags is not None:
            output_tags = ", ".join(output_tags)
            if print_tags:
                    print("trainedWords:",output_tags)
        else:
            output_tags = ""

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if (query_tags and output_tags == "") or force_fetch:
            print("calculating lora hash")
            LORAsha256 = calculate_sha256(lora_path)
            print("requesting infos")
            model_info = get_model_version_info(LORAsha256)
            if model_info is not None:
                if "trainedWords" in model_info:
                    print("tags found!")
                    if lora_tags is None:
                        lora_tags = {}
                    lora_tags[lora_name] = model_info["trainedWords"]
                    save_dict_to_json(lora_tags,json_tags_path)
                    output_tags = ", ".join(model_info["trainedWords"])
                    if print_tags:
                        print("trainedWords:",output_tags)
            else:
                print("No informations found.")
                if lora_tags is None:
                        lora_tags = {}
                lora_tags[lora_name] = []
                save_dict_to_json(lora_tags,json_tags_path)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        if opt_prompt is not None:
            if tags_out:
                output_tags = opt_prompt+", "+output_tags
            else:
                output_tags = opt_prompt
        return (model_lora, clip_lora, output_tags,)
    
NODE_CLASS_MAPPINGS = {
    "LoraLoaderTagsQuery": LoraLoaderTagsQuery,
    "Fetch Lora Tags": FetchLoraTags,
    "Lora Selector": LoraSelector,
}
