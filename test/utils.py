import json

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





