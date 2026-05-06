import json
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub
import numpy as np
import pandas as pd


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def valid_pair(question: str, answer: str) -> bool:
    q = normalize(question)
    a = (answer or "").strip()
    if len(q) < 6 or len(a) < 8:
        return False
    if q == a.lower():
        return False
    if any(bad in q for bad in ["http://", "https://", "www."]):
        return False
    return True


def detailed_answer(question: str, answer: str) -> str:
    answer = (answer or "").strip()
    if len(answer.split()) >= 30:
        return answer
    return (
        f"{answer}\n\n"
        "Detailed explanation: This answer is based on curated training data and is intended to give a practical, easy-to-understand response. "
        "If you want, ask a follow-up and I will break this down step by step with examples."
    )


def load_dialog_pairs(dialog_path: Path, max_rows: int) -> List[Tuple[str, str]]:
    lines = dialog_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pairs: List[Tuple[str, str]] = []

    for line in lines:
        chunks = re.split(r"\t|\s{2,}", line.strip())
        chunks = [c.strip() for c in chunks if c.strip()]
        if len(chunks) < 2:
            continue

        q, a = chunks[0], chunks[1]
        if not valid_pair(q, a):
            continue

        # Prefer questions/informational turns to avoid noisy chit-chat pairs.
        if "?" not in q and len(q.split()) < 4:
            continue

        pairs.append((q, a))
        if len(pairs) >= max_rows:
            break

    return pairs


def load_qa_pairs(csv_path: Path, max_rows: int) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Unexpected CSV schema in {csv_path}")

    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        q = str(row.get("input", ""))
        a = str(row.get("target", ""))
        if not valid_pair(q, a):
            continue
        pairs.append((q.strip(), a.strip()))
        if len(pairs) >= max_rows:
            break

    return pairs


def to_intents(pairs: List[Tuple[str, str]], prefix: str) -> List[Dict]:
    seen = set()
    intents = []

    for i, (q, a) in enumerate(pairs, start=1):
        nq = normalize(q)
        if nq in seen:
            continue
        seen.add(nq)

        intents.append(
            {
                "tag": f"{prefix}_{i}",
                "patterns": [q, nq],
                "responses": [detailed_answer(q, a)],
            }
        )

    return intents


def merge_into_intents_file(intents_file: Path, new_intents: List[Dict]) -> int:
    if intents_file.exists():
        data = json.loads(intents_file.read_text(encoding="utf-8"))
    else:
        data = {"intents": []}

    intents = data.setdefault("intents", [])

    # Remove previously generated Kaggle intents, then re-add fresh.
    intents = [
        it
        for it in intents
        if not str(it.get("tag", "")).startswith("kaggle_qa_")
        and not str(it.get("tag", "")).startswith("kaggle_dialog_")
    ]

    intents.extend(new_intents)
    data["intents"] = intents
    intents_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(new_intents)


def train_neural_model(neural_intents_path: Path, model_dir: Path, epochs: int = 40) -> None:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore[import-not-found]
        from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
        from tensorflow.keras.optimizers import Adam  # type: ignore[import-not-found]
    except Exception:
        from keras.layers import Dense, Dropout, Input
        from keras.models import Sequential
        from keras.optimizers import Adam

    lemmatizer = WordNetLemmatizer()

    for pkg, location in [("punkt", "tokenizers/punkt"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(pkg, quiet=True)

    data = json.loads(neural_intents_path.read_text(encoding="utf-8"))

    words = []
    classes = []
    documents = []
    ignore_tokens = {"?", "!", ",", ".", "'s"}

    for intent in data.get("intents", []):
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        if not tag or not patterns:
            continue

        for pattern in patterns:
            tokens = nltk.word_tokenize(str(pattern))
            words.extend(tokens)
            documents.append((tokens, tag))

        if tag not in classes:
            classes.append(tag)

    words = sorted({lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_tokens})
    classes = sorted(set(classes))

    if not words or not classes or not documents:
        raise RuntimeError("Insufficient training data after preprocessing")

    training = []
    output_empty = [0] * len(classes)

    for token_list, tag in documents:
        token_list = [lemmatizer.lemmatize(w.lower()) for w in token_list]
        bag = [1 if w in token_list else 0 for w in words]

        output_row = list(output_empty)
        output_row[classes.index(tag)] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    x_train = np.array(list(training[:, 0]))
    y_train = np.array(list(training[:, 1]))

    model = Sequential(
        [
            Input(shape=(len(x_train[0]),)),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(len(y_train[0]), activation="softmax"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=1)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "mymodel.keras"))
    model.save(str(model_dir / "mymodel.h5"))

    with open(model_dir / "words.pkl", "wb") as f:
        pickle.dump(words, f)

    with open(model_dir / "classes.pkl", "wb") as f:
        pickle.dump(classes, f)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ai_dir = root / "ai_services"

    dataset_qa = Path(kagglehub.dataset_download("reemmuharram/chatbot-qa-csv")) / "full_dataset.csv"
    dataset_dialogs = Path(kagglehub.dataset_download("grafstor/simple-dialogs-for-chatbot")) / "dialogs.txt"

    qa_pairs = load_qa_pairs(dataset_qa, max_rows=900)
    dialog_pairs = load_dialog_pairs(dataset_dialogs, max_rows=300)

    generated = []
    generated.extend(to_intents(qa_pairs, "kaggle_qa"))
    generated.extend(to_intents(dialog_pairs, "kaggle_dialog"))

    intents_example_path = ai_dir / "intents_example.json"
    neural_intents_path = ai_dir / "neural_model" / "intents.json"

    added_a = merge_into_intents_file(intents_example_path, generated)
    added_b = merge_into_intents_file(neural_intents_path, generated)

    train_neural_model(neural_intents_path, ai_dir / "neural_model", epochs=35)

    report = {
        "qa_pairs_used": len(qa_pairs),
        "dialog_pairs_used": len(dialog_pairs),
        "generated_intents": len(generated),
        "updated_intents_example": str(intents_example_path),
        "updated_neural_intents": str(neural_intents_path),
        "added_to_intents_example": added_a,
        "added_to_neural_intents": added_b,
        "model_dir": str(ai_dir / "neural_model"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
