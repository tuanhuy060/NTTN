# =========================================================
# ĐỒ ÁN NLP - TIỀN XỬ LÍ DATASET SHOPEE SENTIMENT
# Các bước:
# 1. Tải dataset từ Kaggle
# 2. Đọc file chính và file augment
# 3. EDA / kiểm tra dữ liệu
# 4. Làm sạch dữ liệu cơ bản
# 5. Chia train / validation / test
# 6. Thêm augment không dấu vào train
# 7. Tiền xử lí văn bản
# 8. Lưu dữ liệu đã xử lí
# =========================================================

import os
import re
import glob
import unicodedata
import warnings

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =========================================================
# CẤU HÌNH
# =========================================================

DATASET_NAME = "dduongdev/shopee-vietnamese-product-reviews-sentiment"
RANDOM_STATE = 42
OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# IMPORT UNDERTHESEA
# =========================================================

try:
    from underthesea import word_tokenize
    USE_UNDERTHESEA = True
    print("Đã import underthesea thành công.")
except Exception as e:
    USE_UNDERTHESEA = False
    print("Không import được underthesea. Sẽ bỏ qua bước tách từ.")
    print("Lỗi:", e)


# =========================================================
# BƯỚC 1: TẢI DATASET TỪ KAGGLE
# =========================================================

print("\n===== BƯỚC 1: TẢI DATASET TỪ KAGGLE =====")

path = kagglehub.dataset_download(DATASET_NAME)

print("Path to dataset files:", path)

print("\nCác file trong dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))


# =========================================================
# BƯỚC 2: ĐỌC FILE CHÍNH VÀ FILE AUGMENT
# =========================================================

print("\n===== BƯỚC 2: ĐỌC FILE JSONL =====")

jsonl_files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)

print("\nCác file JSONL tìm thấy:")
for f in jsonl_files:
    print(f)

main_candidates = [
    f for f in jsonl_files
    if "shopee_reviews_dataset" in os.path.basename(f)
]

aug_candidates = [
    f for f in jsonl_files
    if "aug_unaccented_reviews" in os.path.basename(f)
]

if not main_candidates:
    raise FileNotFoundError("Không tìm thấy file shopee_reviews_dataset.jsonl")

if not aug_candidates:
    raise FileNotFoundError("Không tìm thấy file aug_unaccented_reviews.jsonl")

main_file = main_candidates[0]
aug_file = aug_candidates[0]

df = pd.read_json(main_file, lines=True)
df_aug = pd.read_json(aug_file, lines=True)

print("\nFile chính:", main_file)
print("File augment:", aug_file)

print("\nKích thước file chính:", df.shape)
print("Kích thước file augment:", df_aug.shape)


# =========================================================
# BƯỚC 3: EDA / KIỂM TRA DỮ LIỆU
# =========================================================

print("\n===== BƯỚC 3: EDA / KIỂM TRA DỮ LIỆU =====")

print("\n5 dòng đầu tiên của file chính:")
print(df.head())

print("\nThông tin dataset chính:")
print(df.info())

print("\nTên các cột:")
print(df.columns.tolist())

print("\nKích thước dữ liệu chính:")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])

print("\nKiểm tra missing value file chính:")
print(df.isnull().sum())

print("\nKiểm tra trùng lặp file chính:")
print("Số ID trùng:", df["id"].duplicated().sum())
print("Số review trùng:", df["review"].duplicated().sum())

print("\nPhân bố label file chính:")
print(df["label"].value_counts())

print("\nTỷ lệ phần trăm label file chính:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nPhân bố rating:")
print(df["rating"].value_counts().sort_index())

print("\nTỷ lệ phần trăm rating:")
print((df["rating"].value_counts(normalize=True).sort_index() * 100).round(2))


# ---------------------------------------------------------
# Kiểm tra độ dài review
# ---------------------------------------------------------

df["review_length"] = df["review"].astype(str).apply(len)
df["word_count"] = df["review"].astype(str).apply(lambda x: len(x.split()))

print("\n===== KIỂM TRA ĐỘ DÀI REVIEW =====")
print("Review có độ dài 0:", (df["review_length"] == 0).sum())
print("Review dưới 5 ký tự:", (df["review_length"] < 5).sum())
print("Review dưới 3 từ:", (df["word_count"] < 3).sum())

print("\nThống kê độ dài review:")
print(df[["review_length", "word_count"]].describe())


# ---------------------------------------------------------
# Kiểm tra rating và label lệch nhau
# ---------------------------------------------------------

print("\n===== KIỂM TRA RATING VÀ LABEL LỆCH NHAU =====")

negative_high_rating = df[
    (df["label"] == "negative") & (df["rating"] >= 4)
]

positive_low_rating = df[
    (df["label"] == "positive") & (df["rating"] <= 3)
]

print("Negative nhưng rating 4-5:", len(negative_high_rating))
print("Positive nhưng rating 1-3:", len(positive_low_rating))

print("\nVí dụ negative nhưng rating cao:")
print(negative_high_rating[["review", "rating", "label"]].head(5))

print("\nVí dụ positive nhưng rating thấp:")
print(positive_low_rating[["review", "rating", "label"]].head(5))


# ---------------------------------------------------------
# Kiểm tra file augment
# ---------------------------------------------------------

print("\n===== KIỂM TRA FILE AUGMENT KHÔNG DẤU =====")

print("\nThông tin file augment:")
print(df_aug.info())

print("\nKích thước file augment:")
print(df_aug.shape)

print("\nPhân bố label trong file augment:")
print(df_aug["label"].value_counts())

print("\nTỷ lệ phần trăm label trong file augment:")
print((df_aug["label"].value_counts(normalize=True) * 100).round(2))

print("\nKiểm tra missing value file augment:")
print(df_aug.isnull().sum())

print("\nSố ID trùng trong augment:")
print(df_aug["id"].duplicated().sum())

print("\nSố review trùng trong augment:")
print(df_aug["review"].duplicated().sum())


# =========================================================
# BƯỚC 4: LÀM SẠCH DỮ LIỆU CƠ BẢN
# =========================================================

print("\n===== BƯỚC 4: LÀM SẠCH DỮ LIỆU CƠ BẢN =====")

df = df.copy()
df_aug = df_aug.copy()

# Ép kiểu id về string để xử lí giao nhau giữa file chính và augment
df["id"] = df["id"].astype(str)
df_aug["id"] = df_aug["id"].astype(str)

# Loại bỏ dòng thiếu review hoặc label
df = df.dropna(subset=["review", "label"]).copy()
df_aug = df_aug.dropna(subset=["review", "label"]).copy()

# Loại bỏ review trùng trong file chính
df = df.drop_duplicates(subset=["review"]).copy()

# Reset index
df = df.reset_index(drop=True)
df_aug = df_aug.reset_index(drop=True)

# Tạo lại cột độ dài sau khi làm sạch
df["review_length"] = df["review"].astype(str).apply(len)
df["word_count"] = df["review"].astype(str).apply(lambda x: len(x.split()))

df_aug["review_length"] = df_aug["review"].astype(str).apply(len)
df_aug["word_count"] = df_aug["review"].astype(str).apply(lambda x: len(x.split()))

print("Dataset chính sau làm sạch:", df.shape)
print("Augment sau làm sạch:", df_aug.shape)

print("\nPhân bố label sau làm sạch:")
print(df["label"].value_counts())


# =========================================================
# BƯỚC 5: CHIA TRAIN / VALIDATION / TEST
# =========================================================

print("\n===== BƯỚC 5: CHIA TRAIN / VALIDATION / TEST =====")

# Chia 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

# Chia temp thành 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)

train_df = train_df.copy().reset_index(drop=True)
val_df = val_df.copy().reset_index(drop=True)
test_df = test_df.copy().reset_index(drop=True)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\nPhân bố label trong train:")
print(train_df["label"].value_counts())
print((train_df["label"].value_counts(normalize=True) * 100).round(2))

print("\nPhân bố label trong validation:")
print(val_df["label"].value_counts())
print((val_df["label"].value_counts(normalize=True) * 100).round(2))

print("\nPhân bố label trong test:")
print(test_df["label"].value_counts())
print((test_df["label"].value_counts(normalize=True) * 100).round(2))

train_ids = set(train_df["id"])
val_ids = set(val_df["id"])
test_ids = set(test_df["id"])

print("\nKiểm tra trùng ID giữa các tập:")
print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Validation ∩ Test:", len(val_ids & test_ids))


# =========================================================
# BƯỚC 6: THÊM AUGMENT KHÔNG DẤU VÀO TRAIN
# =========================================================

print("\n===== BƯỚC 6: THÊM AUGMENT KHÔNG DẤU VÀO TRAIN =====")

# Chỉ lấy augment có id thuộc train
# Không lấy augment của validation/test để tránh leakage
aug_train_df = df_aug[df_aug["id"].isin(train_ids)].copy()
aug_train_df = aug_train_df.reset_index(drop=True)

train_with_aug_df = pd.concat(
    [train_df, aug_train_df],
    ignore_index=True
)

aug_train_ids = set(aug_train_df["id"])

print("Số dòng augment ban đầu:", len(df_aug))
print("Số dòng augment được thêm vào train:", len(aug_train_df))

print("\nTrain gốc:", train_df.shape)
print("Augment thêm vào:", aug_train_df.shape)
print("Train sau augment:", train_with_aug_df.shape)

print("\nPhân bố label train sau augment:")
print(train_with_aug_df["label"].value_counts())
print((train_with_aug_df["label"].value_counts(normalize=True) * 100).round(2))

print("\nKiểm tra leakage augment:")
print("Augment train ∩ Validation:", len(aug_train_ids & val_ids))
print("Augment train ∩ Test:", len(aug_train_ids & test_ids))


# =========================================================
# BƯỚC 7: TIỀN XỬ LÍ VĂN BẢN
# =========================================================

print("\n===== BƯỚC 7: TIỀN XỬ LÍ VĂN BẢN =====")

teen_dict = {
    "ko": "không",
    "k": "không",
    "khong": "không",
    "hok": "không",
    "khum": "không",
    "hong": "không",
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "sp": "sản phẩm",
    "sanpham": "sản phẩm",
    "mn": "mọi người",
    "mng": "mọi người",
    "r": "rồi",
    "roi": "rồi",
    "vs": "với",
    "ok": "ổn",
    "oke": "ổn",
    "oki": "ổn",
    "okie": "ổn",
    "shoppe": "shopee"
}

positive_emojis = [
    "😍", "❤️", "❤", "😊", "👍", "🥰", "😘", "😁", "😄", "😃", "😆", "🤩"
]

negative_emojis = [
    "😡", "😠", "👎", "😭", "😞", "😤", "😢", "☹️", "🙁", "😔"
]


def normalize_unicode(text):
    """
    Chuẩn hóa Unicode tiếng Việt.
    """
    return unicodedata.normalize("NFC", str(text))


def normalize_emoji(text):
    """
    Chuyển emoji thành token cảm xúc.
    """
    for emo in positive_emojis:
        text = text.replace(emo, " emo_pos ")
    for emo in negative_emojis:
        text = text.replace(emo, " emo_neg ")
    return text


def normalize_repeated_chars(text):
    """
    Giảm ký tự lặp quá nhiều.
    Ví dụ: đẹppppp -> đẹpp
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_teencode(text):
    """
    Chuẩn hóa một số teencode và từ viết tắt phổ biến.
    """
    words = text.split()
    normalized_words = []

    for word in words:
        prefix = ""
        suffix = ""

        # Tách ký tự đặc biệt ở đầu từ
        while len(word) > 0 and not word[0].isalnum():
            prefix += word[0]
            word = word[1:]

        # Tách ký tự đặc biệt ở cuối từ
        while len(word) > 0 and not word[-1].isalnum():
            suffix = word[-1] + suffix
            word = word[:-1]

        normalized_word = teen_dict.get(word, word)
        normalized_words.append(prefix + normalized_word + suffix)

    return " ".join(normalized_words)


def remove_noise(text):
    """
    Loại bỏ URL, HTML, ký tự nhiễu.
    Giữ lại:
    - chữ tiếng Việt
    - số
    - khoảng trắng
    - dấu câu cơ bản
    - dấu gạch dưới
    """
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # Giữ chữ tiếng Việt, số, khoảng trắng, dấu câu cơ bản
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_\s.,!?]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    """
    Pipeline tiền xử lí văn bản:
    1. Chuẩn hóa Unicode
    2. Chuyển về chữ thường
    3. Chuẩn hóa emoji
    4. Chuẩn hóa ký tự lặp
    5. Loại bỏ nhiễu
    6. Chuẩn hóa teencode
    7. Tách từ tiếng Việt bằng underthesea nếu có
    """
    text = normalize_unicode(text)
    text = text.lower()
    text = normalize_emoji(text)
    text = normalize_repeated_chars(text)
    text = remove_noise(text)
    text = normalize_teencode(text)

    if USE_UNDERTHESEA:
        try:
            text = word_tokenize(text, format="text")
        except Exception:
            pass

    return text


# Áp dụng tiền xử lí cho các tập dữ liệu
train_df["clean_review"] = train_df["review"].apply(preprocess_text)
val_df["clean_review"] = val_df["review"].apply(preprocess_text)
test_df["clean_review"] = test_df["review"].apply(preprocess_text)
train_with_aug_df["clean_review"] = train_with_aug_df["review"].apply(preprocess_text)

# Có thể tạo thêm test không dấu để dùng sau này nếu cần đánh giá robustness
aug_test_df = df_aug[df_aug["id"].isin(test_ids)].copy()
aug_test_df = aug_test_df.reset_index(drop=True)

if len(aug_test_df) > 0:
    aug_test_df["clean_review"] = aug_test_df["review"].apply(preprocess_text)

print("\nVí dụ trước và sau tiền xử lí:")
for i in range(min(5, len(train_df))):
    print("\nReview gốc:", train_df.loc[i, "review"])
    print("Review sạch:", train_df.loc[i, "clean_review"])


# =========================================================
# BƯỚC 8: LƯU DỮ LIỆU ĐÃ XỬ LÍ
# =========================================================

print("\n===== BƯỚC 8: LƯU DỮ LIỆU ĐÃ XỬ LÍ =====")

# Lưu CSV
train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "val_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

train_with_aug_df.to_csv(
    os.path.join(OUTPUT_DIR, "train_with_aug_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

if len(aug_test_df) > 0:
    aug_test_df.to_csv(
        os.path.join(OUTPUT_DIR, "aug_test_clean.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# Lưu JSONL
train_df.to_json(
    os.path.join(OUTPUT_DIR, "train_clean.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False
)

val_df.to_json(
    os.path.join(OUTPUT_DIR, "val_clean.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False
)

test_df.to_json(
    os.path.join(OUTPUT_DIR, "test_clean.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False
)

train_with_aug_df.to_json(
    os.path.join(OUTPUT_DIR, "train_with_aug_clean.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False
)

if len(aug_test_df) > 0:
    aug_test_df.to_json(
        os.path.join(OUTPUT_DIR, "aug_test_clean.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False
    )


# Lưu thống kê đơn giản
summary = {
    "main_dataset_rows": len(df),
    "augment_dataset_rows": len(df_aug),
    "train_rows": len(train_df),
    "validation_rows": len(val_df),
    "test_rows": len(test_df),
    "aug_train_rows": len(aug_train_df),
    "train_with_aug_rows": len(train_with_aug_df),
    "aug_test_rows": len(aug_test_df),
    "train_label_distribution": train_df["label"].value_counts().to_dict(),
    "validation_label_distribution": val_df["label"].value_counts().to_dict(),
    "test_label_distribution": test_df["label"].value_counts().to_dict(),
    "train_with_aug_label_distribution": train_with_aug_df["label"].value_counts().to_dict(),
    "use_underthesea": USE_UNDERTHESEA
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "data_preprocessing_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu dữ liệu đã xử lí vào thư mục:", OUTPUT_DIR)

print("\nCác file đã lưu:")
for file in os.listdir(OUTPUT_DIR):
    print("-", os.path.join(OUTPUT_DIR, file))

print("\n===== HOÀN THÀNH TIỀN XỬ LÍ DỮ LIỆU =====")