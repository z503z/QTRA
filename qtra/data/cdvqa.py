import json
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms, io

from .simple_tokenizer import SimpleTokenizer as _Tokenizer


w2v_dict = {
    '0': 0,
    '0_to_10': 1,
    '10_to_20': 2,
    '20_to_30': 3,
    '30_to_40': 4,
    '40_to_50': 5,
    '50_to_60': 6,
    '60_to_70': 7,
    '70_to_80': 8,
    '80_to_90': 9,
    '90_to_100': 10,
    'NVG_surface': 11,
    'buildings': 12,
    'low_vegetation': 13,
    'playgrounds': 14,
    'trees': 15,
    'water': 16,
    'no': 17,
    'yes': 18,
    'green_house': 19,
    'road': 20,
    'bridge': 21,
    'others': 22,
}

_tokenizer = _Tokenizer()


def normalize_answer(ans: str) -> str:
    # CDVQA JSON uses names like "0%-10%" and "NVG surface";
    # VisTA dict uses "0_to_10" and "NVG_surface".
    ans = ans.strip()
    ans = ans.replace(" ", "_")
    ans = ans.replace("%-", "_to_")
    ans = ans.replace("%", "")
    return ans


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 40,
    truncate: bool = True,
) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class CDVQADataset(Dataset):
    def __init__(self, cdvqa_root, second_root, split):
        self.cdvqa_root = Path(cdvqa_root)
        self.second_root = Path(second_root)
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            ZeroOneNormalize(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.image_index = self._build_image_index()
        self.samples = self._build_samples()

        print(f"[CDVQA {split}] samples: {len(self.samples)}")

    def _build_image_index(self):
        candidate_roots = [
            self.second_root / "SECOND_train_set",
            self.second_root / "test",
        ]

        index = {}
        for root in candidate_roots:
            im1_dir = root / "im1"
            im2_dir = root / "im2"

            if not im1_dir.exists() or not im2_dir.exists():
                print(f"[WARN] missing im1/im2 in {root}")
                continue

            for p1 in im1_dir.iterdir():
                if p1.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    continue

                name = p1.name
                p2 = im2_dir / name
                if p2.exists():
                    index[name] = {
                        "im1": p1,
                        "im2": p2,
                    }

        return index

    def _build_samples(self):
        with open(self.cdvqa_root / f"{self.split}_images.json", "r") as f:
            images = json.load(f)["images"]

        with open(self.cdvqa_root / f"{self.split}_questions.json", "r") as f:
            questions = json.load(f)["questions"]

        with open(self.cdvqa_root / f"{self.split}_answers.json", "r") as f:
            answers = json.load(f)["answers"]

        image_by_id = {x["id"]: x for x in images}
        answer_by_id = {x["id"]: x for x in answers}

        samples = []
        missing = 0
        unknown_answer = 0

        for q in questions:
            img_info = image_by_id[q["img_id"]]
            file_name = img_info["file_name"]

            if file_name not in self.image_index:
                missing += 1
                continue

            raw_answer = answer_by_id[q["answers_ids"][0]]["answer"]
            answer = normalize_answer(raw_answer)

            if answer not in w2v_dict:
                unknown_answer += 1
                print(f"[WARN] unknown answer: raw={raw_answer}, normalized={answer}")
                continue

            samples.append({
                "file_name": file_name,
                "question": q["question"],
                "type": q["type"],
                "answer_label": w2v_dict[answer],
                "answer_text": answer,
                "qid": q["id"],
            })

        if missing:
            print(f"[WARN] missing image samples: {missing}")
        if unknown_answer:
            print(f"[WARN] unknown answer samples: {unknown_answer}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        paths = self.image_index[s["file_name"]]

        img1 = io.read_image(str(paths["im1"]))
        img2 = io.read_image(str(paths["im2"]))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        q_str = tokenize(s["question"], 40, True).squeeze(0)
        answer_label = torch.tensor(s["answer_label"], dtype=torch.long)

        return img1, img2, q_str, s["type"], answer_label, s["file_name"]
