# Evaluation Script for Qwen2.5-VL-7B-Instruct on CORD-v2 Test Set using qwen_vl_utils

import torch
import json
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from datasets import Dataset, Features, Value, Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration

from utils import clean_output, extract_fields, compute_fieldwise_accuracy


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

print("Starting loading model...")
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
print("Finished loading model...")

cord_dataset = load_dataset("naver-clova-ix/cord-v2")
test_set = cord_dataset["test"]

print("Starting evaluation...")
records = []
total_correct = 0
total_elements = 0

with tqdm(enumerate(test_set), total=len(test_set), desc="Evaluating") as pbar:
    for idx, sample in pbar:
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
            "ground_truth": json.dumps(true_fields, ensure_ascii=False),
            "predicted": json.dumps(pred_fields, ensure_ascii=False),
            "accuracy": float(accuracy),
            "correct": int(correct),
            "total": int(total)
        })

        total_correct += correct
        total_elements += total

        running_avg = (total_correct / total_elements) if total_elements else 0.0
        pbar.set_description(f"[{idx + 1:04d}] Accuracy: {accuracy:.4f} ({correct}/{total})")
        pbar.set_postfix(running_avg=f"{running_avg:.4f}", correct=total_correct, total=total_elements)
print("Finished evaluation...")

features = Features({
    "id": Value("int32"),
    "image": Image(decode=True, id=None),
    "ground_truth": Value("string", id=None),
    "predicted": Value("string", id=None),
    "accuracy": Value("float32", id=None),
    "correct": Value("int32", id=None),
    "total": Value("int32", id=None)
})
pred_dataset = Dataset.from_list(records, features=features)
pred_dataset.save_to_disk("datasets/cord_v2_qwen2.5-vl-3b")

# from datasets import load_from_disk
# ds = load_from_disk("datasets/cord_v2_qwen2.5-vl-3b")


overall_accuracy = total_correct / total_elements if total_elements else 0
print(f"Field-wise Accuracy: {overall_accuracy:.4f}")
print(f"++++++++++++++++++++++++++++++++++++")
