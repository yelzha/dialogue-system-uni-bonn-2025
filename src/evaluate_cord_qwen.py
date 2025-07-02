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
    return text.strip()

def extract_fields(record_raw):
    record_json = json.loads(record_raw)
    json_data = record_json['gt_parse']
    fields = {}
    for key in ["menu", "sub_total", "total"]:
        if key in json_data:
            fields[key] = json_data[key]
    return fields



predictions, ground_truths = [], []

instruction = (
    "Extract and return all structured fields from this receipt image in the following JSON format. "
    "Make sure to follow the correct hierarchical structure and include all known keys from the dataset:\n"
    "{\n"
    "  \"menu\": {\n"
    "    \"nm\": string,\n"
    "    \"num\": string,\n"
    "    \"cnt\": string,\n"
    "    \"price\": string,\n"
    "    \"itemsubtotal\": string,\n"
    "    \"sub_nm\": optional string,\n"
    "    \"sub_cnt\": optional string,\n"
    "    \"sub_price\": optional string\n"
    "  },\n"
    "  \"sub_total\": {\n"
    "    \"subtotal_price\": string,\n"
    "    \"discount_price\": string,\n"
    "    \"tax_price\": string\n"
    "  },\n"
    "  \"total\": {\n"
    "    \"total_price\": string,\n"
    "    \"creditcardprice\": string,\n"
    "    \"menuqty_cnt\": string\n"
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

# Basic evaluation using exact match comparison
correct = sum([p == t for p, t in zip(predictions, ground_truths)])
total = len(predictions)
accuracy = correct / total if total else 0

print(f"Evaluation Results:")
print(f"Total Samples Evaluated: {total}")
print(f"Exact Match Accuracy: {accuracy:.4f}")
