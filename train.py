# Copyright (C) 2026 Dang2nek
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# ⚡ CẤU HÌNH
# ============================================================
BASE_DIR     = Path(__file__).parent
DATA_PATH    = BASE_DIR / "data_blhd.csv"
OUTPUT_DIR   = str(BASE_DIR / "blhd_model")
MODEL_NAME   = "vinai/phobert-base"
MAX_LENGTH   = 64
BATCH_SIZE   = 8
EPOCHS       = 5
LR           = 3e-5          # learning rate cho lần train gốc
LR_INCREMENTAL = 2e-5        # nhỏ hơn khi train thêm → tránh quên kiến thức cũ
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 10           # đóng băng 10/12 layer
# ============================================================

# ── 1. Tối ưu CPU threads ────────────────────────────────────
num_threads = os.cpu_count() or 4
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(max(1, num_threads // 2))
torch.backends.opt_einsum.enabled = True

print(f"✅ iGPU mode → chạy trên CPU host")
print(f"   Threads  : {torch.get_num_threads()} (inter: {torch.get_num_interop_threads()})")
print(f"   PyTorch  : {torch.__version__}")
print(f"   Data     : {DATA_PATH}")

DEVICE      = torch.device("cpu")
USE_CPU_ARG = True


# ════════════════════════════════════════════════════════════
# HÀM DÙNG CHUNG
# ════════════════════════════════════════════════════════════

def freeze_layers(model, n_freeze=FREEZE_LAYERS):
    """Đóng băng n_freeze layers đầu tiên của encoder."""
    for i, layer in enumerate(model.roberta.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = (i >= n_freeze)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🔒 Đóng băng {n_freeze}/12 layers")
    print(f"   Tham số train: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def build_tokenizer_and_collator(model_path):
    tokenizer     = AutoTokenizer.from_pretrained(str(model_path))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    return tokenizer, data_collator


def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1"      : round(f1_score(labels, preds, average="macro"), 4),
    }


def make_training_args(output_dir, epochs, lr, extra_tag=""):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        use_cpu=USE_CPU_ARG,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        
    )


def evaluate_and_print(trainer, eval_df):
    print("\n📈 Đánh giá trên tập eval:")
    results = trainer.evaluate()
    print(f"   Accuracy : {results['eval_accuracy']}")
    print(f"   F1 Macro : {results['eval_f1']}")

    raw_pred    = trainer.predict(trainer.eval_dataset)
    final_preds = np.argmax(raw_pred.predictions, axis=-1)
    print("\n" + classification_report(
        eval_df["label"].values,
        final_preds,
        target_names=["Không BLHĐ", "BLHĐ"],
        digits=4,
    ))
    return results


def predict(text, model, tokenizer, threshold=0.5):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH, padding=True,
    )
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    label = 1 if probs[1] >= threshold else 0
    return {
        "label"     : "🔴 BLHĐ" if label == 1 else "🟢 Bình thường",
        "p_blhd"    : f"{probs[1]*100:.1f}%",
        "p_normal"  : f"{probs[0]*100:.1f}%",
        "confidence": f"{max(probs)*100:.1f}%",
    }


# ════════════════════════════════════════════════════════════
# CHẾ ĐỘ 1 — TRAIN GỐC (từ đầu)
# ════════════════════════════════════════════════════════════

def train_from_scratch():
    print("\n" + "="*55)
    print("   🏋️  TRAIN GỐC — từ PhoBERT pretrained")
    print("="*55)

    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    print(f"\n📊 Dataset: {len(df)} mẫu")
    print(df["label"].value_counts().rename({0:"Không BLHĐ", 1:"BLHĐ"}).to_string())

    train_df, eval_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    print(f"\n   Train: {len(train_df)} | Eval: {len(eval_df)}")

    tokenizer, data_collator = build_tokenizer_and_collator(MODEL_NAME)
    train_dataset = tokenize_dataset(train_df, tokenizer)
    eval_dataset  = tokenize_dataset(eval_df,  tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(DEVICE)
    freeze_layers(model, FREEZE_LAYERS)

    training_args = make_training_args(OUTPUT_DIR, EPOCHS, LR)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n🚀 Bắt đầu train...\n")
    train_result = trainer.train()
    print(f"\n⏱️  Thời gian : {train_result.metrics['train_runtime']:.0f}s")
    print(f"   Tốc độ    : {train_result.metrics['train_samples_per_second']:.1f} samples/s")

    evaluate_and_print(trainer, eval_df)

    model.save_pretrained(OUTPUT_DIR, save_safetensors=False,)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model lưu tại: {OUTPUT_DIR}/")

    return model, tokenizer


# ════════════════════════════════════════════════════════════
# CHẾ ĐỘ 2 — TRAIN THÊM (incremental)
# ════════════════════════════════════════════════════════════

def load_incremental_data():
    """Hỏi người dùng nguồn dữ liệu train thêm."""
    print("\n📥 Nguồn dữ liệu train thêm:")
    print("   [1] Nhập câu thủ công")
    print("   [2] Load từ file CSV")

    choice = input("\nChọn (1/2): ").strip()

    if choice =="2":
    # Load từ nhiều file CSV
        all_dfs = []
        print("Nhập đường dẫn từng file CSV. Gõ 'xong' khi đã nhập đủ.\n")

        while True:
            csv_path = input(f"  File {len(all_dfs)+1} (hoặc 'xong'): ").strip().strip('"')

            if csv_path.lower() in ("xong", "done", "q", ""):
                if len(all_dfs) == 0:
                    print("⚠️  Chưa nhập file nào!")
                    continue
                break

            path = Path(csv_path)
            if not path.exists():
                print(f"  ❌ Không tìm thấy: {csv_path}")
                continue

            df_tmp = pd.read_csv(path)
            if "text" not in df_tmp.columns or "label" not in df_tmp.columns:
                print("  ❌ CSV cần có cột 'text' và 'label'")
                continue

            df_tmp = df_tmp[["text", "label"]].dropna()
            df_tmp["label"] = df_tmp["label"].astype(int)
            all_dfs.append(df_tmp)
            print(f"  ✅ {path.name}: {len(df_tmp)} mẫu (BLHĐ: {df_tmp['label'].sum()} · Bình thường: {(df_tmp['label']==0).sum()})")

        # Gộp tất cả lại, xáo trộn
        df = pd.concat(all_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\n📊 Tổng cộng {len(df)} mẫu từ {len(all_dfs)} file:")
        print(df["label"].value_counts().rename({0:"Không BLHĐ", 1:"BLHĐ"}).to_string())
        return df

    else:
        # Nhập thủ công
        samples = []
        print("\nNhập câu văn và nhãn. Gõ 'xong' để kết thúc.\n")

        while True:
            text = input(f"  Câu {len(samples)+1}: ").strip()
            if text.lower() in ("xong", "done", "q", ""):
                if len(samples) < 2:
                    print("⚠️  Cần ít nhất 2 mẫu!")
                    continue
                break

            while True:
                lbl = input(f"    Nhãn (1=BLHĐ / 0=Bình thường): ").strip()
                if lbl in ("0", "1"):
                    samples.append({"text": text, "label": int(lbl)})
                    n_blhd   = sum(1 for s in samples if s["label"] == 1)
                    n_normal = sum(1 for s in samples if s["label"] == 0)
                    print(f"    ✅ Đã thêm | Tổng: {len(samples)} (BLHĐ: {n_blhd} · Bình thường: {n_normal})")
                    break
                print("    ⚠️  Nhập 0 hoặc 1")

        df = pd.DataFrame(samples)
        print(f"\n📊 Tổng {len(df)} mẫu mới:")
        print(df["label"].value_counts().rename({0:"Không BLHĐ", 1:"BLHĐ"}).to_string())
        return df


def train_incremental():
    print("\n" + "="*55)
    print("   ➕  TRAIN THÊM — từ model đã có")
    print("="*55)

    model_path = Path(OUTPUT_DIR)
    if not model_path.exists():
        print(f"\n❌ Chưa có model tại {OUTPUT_DIR}")
        print("   Hãy chọn [1] để train gốc trước.")
        return None, None

    # Load model đã train
    print(f"\n⏳ Load model từ: {model_path}")
    tokenizer, data_collator = build_tokenizer_and_collator(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), num_labels=2)
    model = model.to(DEVICE)
    print("✅ Load xong!")

    # Load dữ liệu mới
    new_df = load_incremental_data()

    # Số epochs
    epochs_input = input(f"\nSố epochs (mặc định 3, khuyến nghị 2-5): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 3
    epochs = max(1, min(20, epochs))

    # Split nếu đủ dữ liệu, nếu không train toàn bộ
    if len(new_df) >= 10:
        train_df, eval_df = train_test_split(
            new_df, test_size=0.2, random_state=42, stratify=new_df["label"]
        )
        print(f"\n   Train: {len(train_df)} | Eval: {len(eval_df)}")
        has_eval = True
    else:
        train_df = new_df
        eval_df  = new_df   # dùng luôn toàn bộ để eval khi ít mẫu
        print(f"\n   Train: {len(train_df)} mẫu (ít mẫu → không chia eval)")
        has_eval = False

    train_dataset = tokenize_dataset(train_df, tokenizer)
    eval_dataset  = tokenize_dataset(eval_df,  tokenizer)

    # Unfreeze 2 layer cuối + classifier (LR nhỏ hơn để không quên kiến thức cũ)
    freeze_layers(model, FREEZE_LAYERS)

    training_args = make_training_args(OUTPUT_DIR, epochs, LR_INCREMENTAL)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n🚀 Bắt đầu train thêm ({len(train_df)} mẫu · {epochs} epochs)...\n")
    train_result = trainer.train()
    print(f"\n⏱️  Thời gian : {train_result.metrics['train_runtime']:.0f}s")
    print(f"   Tốc độ    : {train_result.metrics['train_samples_per_second']:.1f} samples/s")

    if has_eval:
        evaluate_and_print(trainer, eval_df)
    else:
        results = trainer.evaluate()
        print(f"\n📈 Kết quả: Accuracy={results['eval_accuracy']} · F1={results['eval_f1']}")

    # Lưu đè lên model cũ
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model đã cập nhật tại: {OUTPUT_DIR}/")

    return model, tokenizer


# ════════════════════════════════════════════════════════════
# TEST THỬ SAU KHI TRAIN
# ════════════════════════════════════════════════════════════

def run_test(model, tokenizer):
    print("\n" + "="*55)
    print("   🧪  TEST THỬ")
    print("="*55)

    default_tests = [
        "THẰNG MẶT LỒN, GẶP TAO Ở CỔNG TRƯỜNG, TAO SẼ CHO MÀY MỘT TRẬN RA TRÒ",
        "Cả băng tao chặn đường thằng đó, đánh cho nó một trận.",
        "Hôm nay trời đẹp, cả lớp đi dã ngoại vui lắm.",
        "Bạn ơi, cho mình hỏi bài tập toán với nhé?",
        "Mày dám nộp đơn tố cáo tao hả, mày không sợ à.",
    ]

    print("\nCâu test mặc định:")
    for t in default_tests:
        r = predict(t, model, tokenizer)
        print(f"  {r['label']} ({r['p_blhd']}) | {t}")

    print("\nNhập câu tùy ý để test. Gõ 'q' để thoát.")
    while True:
        text = input("\n📝 Câu: ").strip()
        if text.lower() in ("q", "quit", "thoat", ""):
            break
        r = predict(text, model, tokenizer)
        print(f"   {r['label']}")
        print(f"   Xác suất BLHĐ    : {r['p_blhd']}")
        print(f"   Xác suất bình thường: {r['p_normal']}")
        print(f"   Độ tin cậy       : {r['confidence']}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model_exists = Path(OUTPUT_DIR).exists()

    print("\n" + "="*55)
    print("   🛡️  HỆ THỐNG HUẤN LUYỆN PHÁT HIỆN BLHĐ")
    print("="*55)

    if model_exists:
        print(f"\n✅ Đã tìm thấy model tại: {OUTPUT_DIR}/")
        print("\n   [1] Train thêm (cập nhật model hiện có)")
        print("   [2] Train lại từ đầu (ghi đè model cũ)")
        print("   [3] Chỉ test thử model hiện có")

        choice = input("\nChọn (1/2/3): ").strip()
    else:
        print(f"\n⚠️  Chưa có model tại {OUTPUT_DIR}/")
        print("   Cần train gốc trước.\n")
        choice = "2"

    # ── Thực hiện theo lựa chọn ──────────────────────────────
    if choice == "1":
        model, tokenizer = train_incremental()

    elif choice == "2":
        confirm = input("\n⚠️  Train lại từ đầu sẽ GHI ĐÈ model cũ. Tiếp tục? (y/n): ").strip().lower()
        if confirm == "y":
            model, tokenizer = train_from_scratch()
        else:
            print("Đã hủy.")
            exit()

    elif choice == "3":
        print(f"\n⏳ Load model từ: {OUTPUT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
        model.eval()
        print("✅ Load xong!")

    else:
        print("Lựa chọn không hợp lệ.")
        exit()

    # ── Test thử sau khi train ────────────────────────────────
    if model is not None:
        do_test = input("\n🧪 Test thử model ngay? (y/n, mặc định y): ").strip().lower()
        if do_test != "n":
            run_test(model, tokenizer)

    print("\n👋 Hoàn tất!")

