# Evaluation Script for Qwen2.5-VL-7B-Instruct on CORD-v2 Test Set using qwen_vl_utils

import torch
import json
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration

# Load model and processor
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Load dataset
cord_dataset = load_dataset("naver-clova-ix/cord-v2")
test_set = cord_dataset["test"]

def clean_output(text):
    text = text.strip()

    # Convert escaped characters to real ones
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass  # fallback in case decode fails

    # Remove code block markers if present
    if text.startswith("```"):
        parts = text.split("\n", 1)
        if len(parts) == 2:
            text = parts[1]  # remove the first line like ```json
        if text.endswith("```"):
            text = text[:-3]  # remove the closing ```

    return text.strip()

def extract_fields(record_raw):
    record_json = json.loads(record_raw)
    json_data = record_json['gt_parse']
    fields = {}
    for key in ["menu", "sub_total", "total"]:
        if key in json_data:
            fields[key] = json_data[key]
    return fields



def normalize_menu(menu):
    if isinstance(menu, dict):
        return [menu]
    elif isinstance(menu, list):
        return menu
    return []

def normalize_dict(d):
    return {k: str(v).strip() for k, v in d.items() if v is not None}

def fields_equal(pred, true):
    if not isinstance(pred, dict) or not isinstance(true, dict):
        return False

    pred_menu = normalize_menu(pred.get("menu", []))
    true_menu = normalize_menu(true.get("menu", []))

    if len(pred_menu) != len(true_menu):
        return False

    for p_item, t_item in zip(pred_menu, true_menu):
        p_item = normalize_dict(p_item)
        t_item = normalize_dict(t_item)
        for key in t_item:
            if p_item.get(key, "") != t_item[key]:
                return False

    for section in ["sub_total", "total"]:
        p_sec = normalize_dict(pred.get(section, {}))
        t_sec = normalize_dict(true.get(section, {}))
        for key in t_sec:
            if p_sec.get(key, "") != t_sec[key]:
                return False

    return True


predictions, ground_truths = [], []

instruction = (
    "Extract and return all structured fields from this receipt image in the following JSON format. "
    "Make sure to follow the correct hierarchical structure and include all known keys from the dataset:\n"
    "{\n"
    "  \"menu\": [\n"
    "    {\n"
    "      \"nm\": string,\n"
    "      \"num\": string,\n"
    "      \"cnt\": string,\n"
    "      \"price\": string,\n"
    "      \"itemsubtotal\": string,\n"
    "      \"sub_nm\": optional string,\n"
    "      \"sub_cnt\": optional string,\n"
    "      \"sub_price\": optional string\n"
    "    }\n"
    "  ],\n"
    "  \"sub_total\": {\n"
    "    \"subtotal_price\": string,\n"
    "    \"discount_price\": string,\n"
    "    \"tax_price\": string,\n"
    "    \"service_price\": string,\n"
    "    \"etc\": string\n"
    "  },\n"
    "  \"total\": {\n"
    "    \"total_price\": string,\n"
    "    \"cashprice\": string,\n"
    "    \"changeprice\": string,\n"
    "    \"creditcardprice\": string,\n"
    "    \"emoneyprice\": string,\n"
    "    \"menutype_cnt\": string,\n"
    "    \"menuqty_cnt\": string,\n"
    "    \"total_etc\": string\n"
    "  }\n"
    "}\n"
    "Output only the JSON. Do not include explanations or other text."
)

print("Starting evaluation...")

for sample in tqdm(test_set, desc="Evaluating"):
    image = sample["image"]
    label_json = sample["ground_truth"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    pred_result = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    pred_result = clean_output(pred_result)

    try:
        pred_fields = extract_fields(json.dumps({"gt_parse": json.loads(pred_result)}))
    except:
        pred_fields = {}

    true_fields = extract_fields(label_json)

    for field in true_fields:
        predictions.append(pred_fields.get(field, ""))
        ground_truths.append(true_fields[field])

# Use this for evaluation
correct = sum([fields_equal(p, t) for p, t in zip(predictions, ground_truths)])
total = len(predictions)
accuracy = correct / len(predictions) if predictions else 0
print(f"Accuracy: {accuracy:.4f}")

print(f"Evaluation Results:")
print(f"Total Samples Evaluated: {total}")
print(f"Exact Match Accuracy: {accuracy:.4f}")
