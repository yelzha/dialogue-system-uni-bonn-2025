# Evaluation Script for Qwen2.5-VL-7B-Instruct on CORD-v2 Test Set using qwen_vl_utils

import torch
import json
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

cord_dataset = load_dataset("naver-clova-ix/cord-v2")
test_set = cord_dataset["test"]

def clean_output(text):
    text = text.strip()
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass

    if text.startswith("```"):
        parts = text.split("\n", 1)
        if len(parts) == 2:
            text = parts[1]
        if text.endswith("```"):
            text = text[:-3]

    return text.strip()

def extract_fields(record_raw):
    record_json = json.loads(record_raw)
    json_data = record_json['gt_parse']
    fields = {}
    for key in ["menu", "sub_total", "total"]:
        if key in json_data:
            if key == "menu":
                fields[key] = normalize_menu(json_data[key])
            else:
                fields[key] = json_data[key]
    return fields

def normalize_menu(menu):
    if isinstance(menu, dict):
        return [normalize_dict(menu)]
    elif isinstance(menu, list):
        return [normalize_dict(m) for m in menu]
    return []

def normalize_dict(d):
    return {k: str(v).strip() for k, v in d.items() if v is not None}

def compute_fieldwise_accuracy(pred_fields, true_fields):
    correct = 0
    total = 0

    # Handle menu
    pred_menu = normalize_menu(pred_fields.get("menu", []))
    true_menu = normalize_menu(true_fields.get("menu", []))

    for t_item in true_menu:
        total += len(t_item)
        matched = False
        for p_item in pred_menu:
            # Check if all keys in true item match with predicted
            if all(p_item.get(k, "").strip() == v.strip() for k, v in t_item.items()):
                correct += len(t_item)
                matched = True
                break
        if not matched:
            correct += sum(1 for k in t_item if any(p_item.get(k, "") == t_item[k] for p_item in pred_menu))

    # Handle sub_total and total
    for section in ["sub_total", "total"]:
        pred_section = normalize_dict(pred_fields.get(section, {}))
        true_section = normalize_dict(true_fields.get(section, {}))

        for k, v in true_section.items():
            total += 1
            if pred_section.get(k, "") == v:
                correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total

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

records = []
total_correct = 0
total_elements = 0

for idx, sample in enumerate(test_set):
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

    accuracy, correct, total = compute_fieldwise_accuracy(pred_fields, true_fields)

    records.append({
        "index": idx,
        "image": image,
        "ground_truth": true_fields,
        "predicted": pred_fields,
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    })
    total_correct += correct
    total_elements += total

    print((
        f"[{idx + 1:04d}] Accuracy: {accuracy:.4f} ({correct}/{total}) | "
        f"Running Avg: {(total_correct / total_elements):.4f} "
        f"({total_correct}/{total_elements})"
    ))


pred_dataset = Dataset.from_list(records)
pred_dataset.save_to_disk("datasets/cord_v2_qwen2.5-vl-3b")

# from datasets import load_from_disk
# ds = load_from_disk("datasets/cord_v2_qwen2.5-vl-3b")


overall_accuracy = total_correct / total_elements if total_elements else 0
print(f"Field-wise Accuracy: {overall_accuracy:.4f}")
print(f"++++++++++++++++++++++++++++++++++++")
