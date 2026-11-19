import streamlit as st
import json
import random
import os
import time
import hashlib
import secrets
import fcntl
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pandas import DataFrame, read_csv, concat

@contextmanager
def file_lock(target_path):
    """Cross-session lock so that two people saving at the same moment (e.g. many
    participants finishing a quiz category together) don't clobber each other's data.
    Locks on a sibling '<file>.lock' file; the actual read-modify-write of the real
    file should happen entirely inside the 'with' block."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    lock_path = target_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

# Path to this file's own folder, so the app works no matter where it's deployed
# (local Mac, Streamlit Community Cloud, etc.) instead of a hardcoded absolute path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def resolve_image_path(path):
    """Local image paths in the question JSON are relative (e.g. 'images/onsen.jpeg');
    resolve them against this app's own folder. Remote http(s) URLs pass through unchanged."""
    if not path or path.startswith("http://") or path.startswith("https://"):
        return path
    return os.path.join(BASE_DIR, path)

# ---------- Nationality / language options (for the searchable select boxes on the registration form) ----------
COUNTRIES = ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia, Plurinational State of", "Bonaire, Sint Eustatius and Saba", "Bosnia and Herzegovina", "Botswana", "Bouvet Island", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Congo, The Democratic Republic of the", "Cook Islands", "Costa Rica", "Croatia", "Cuba", "Curaçao", "Cyprus", "Czechia", "Côte d'Ivoire", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "French Guiana", "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heard Island and McDonald Islands", "Holy See (Vatican City State)", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran, Islamic Republic of", "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica", "Japan", "Jersey", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, Democratic People's Republic of", "Korea, Republic of", "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macao", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia, Federated States of", "Moldova, Republic of", "Monaco", "Mongolia", "Montenegro", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "North Macedonia", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Palestine, State of", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russian Federation", "Rwanda", "Réunion", "Saint Barthélemy", "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin (French part)", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten (Dutch part)", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan, Province of China", "Tajikistan", "Tanzania, United Republic of", "Thailand", "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Türkiye", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "United States Minor Outlying Islands", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela, Bolivarian Republic of", "Viet Nam", "Virgin Islands, British", "Virgin Islands, U.S.", "Wallis and Futuna", "Western Sahara", "Yemen", "Zambia", "Zimbabwe", "Åland Islands", "Other"]

REASON_CATEGORY_ORDER = [
    "rule_based", "formality_assumption", "empathy_based", "achievement_bias",
    "symbolic", "indulgence_assumption", "practical", "own_culture", "no_reason", "other",
]

# Reason-option labels, localized. Falls back to English for any language/key not listed.
# Each category maps to one of Hofstede's 6 dimensions (or is a non-dimension control
# category), kept in sync with bucket() in the Research tab:
#   rule_based            -> UAI (uncertainty avoidance)
#   formality_assumption  -> PDI (power distance)           [added]
#   empathy_based         -> IDV (individualism/collectivism)
#   achievement_bias      -> MAS (masculinity/achievement)  [added]
#   symbolic              -> LTO (long-term orientation/tradition)
#   indulgence_assumption -> IVR (indulgence/restraint)     [added]
#   practical             -> (no dimension; neutral/functional control)
#   own_culture            -> (no dimension; own-culture transfer, central to RQ2)
REASON_LABELS = {
    "en": {
        "rule_based": "I heard it's a fixed rule",
        "formality_assumption": "I thought it was about respecting someone of higher status",
        "practical": "I thought it was for cleanliness/safety",
        "empathy_based": "To avoid being rude to others",
        "achievement_bias": "I thought it was about showing skill or doing things properly",
        "symbolic": "I thought it had religious/traditional meaning",
        "indulgence_assumption": "I thought there were fewer restrictions since it's a relaxed/special occasion",
        "own_culture": "We do the same thing in my country",
        "no_reason": "Just a guess / gut feeling",
        "other": "Other (please specify)",
    },
    "ja": {
        "rule_based": "決まったルールだと聞いたから",
        "formality_assumption": "目上の人・立場が上の人への配慮だと思ったから",
        "practical": "清潔・安全のためだと思ったから",
        "empathy_based": "他人に失礼にならないようにするためだと思ったから",
        "achievement_bias": "きちんとできていることや努力を示すためだと思ったから",
        "symbolic": "宗教的・伝統的な意味があると思ったから",
        "indulgence_assumption": "特別な/くつろいだ場だから、普段より制限が少ないと思ったから",
        "own_culture": "自分の国でも同じことをするから",
        "no_reason": "特に理由はなく、何となく",
        "other": "その他（自由記述）",
    },
    "th": {
        "rule_based": "ได้ยินมาว่าเป็นกฎตายตัว",
        "formality_assumption": "คิดว่าเป็นการให้ความเคารพผู้ที่มีสถานะสูงกว่า",
        "practical": "คิดว่าเพื่อความสะอาด/ความปลอดภัย",
        "empathy_based": "เพื่อไม่ให้เสียมารยาทต่อผู้อื่น",
        "achievement_bias": "คิดว่าเป็นการแสดงความสามารถหรือความตั้งใจทำสิ่งต่างๆ ให้ถูกต้อง",
        "symbolic": "คิดว่ามีความหมายทางศาสนา/ประเพณี",
        "indulgence_assumption": "คิดว่ามีข้อจำกัดน้อยลงเพราะเป็นโอกาสพิเศษ/ผ่อนคลาย",
        "own_culture": "ประเทศของฉันก็ทำแบบเดียวกัน",
        "no_reason": "แค่เดา / รู้สึกเฉยๆ",
        "other": "อื่นๆ (โปรดระบุ)",
    },
    "zh": {
        "rule_based": "听说这是固定规则",
        "formality_assumption": "我以为是为了尊重地位更高的人",
        "practical": "我以为是为了清洁/安全",
        "empathy_based": "为了不对他人失礼",
        "achievement_bias": "我以为是为了展现能力或做事认真",
        "symbolic": "我以为有宗教/传统意义",
        "indulgence_assumption": "我以为因为是特别/轻松的场合，限制会比较少",
        "own_culture": "我的国家也会这样做",
        "no_reason": "只是猜的 / 凭感觉",
        "other": "其他（请说明）",
    },
    "ko": {
        "rule_based": "정해진 규칙이라고 들었어요",
        "formality_assumption": "윗사람이나 지위가 높은 사람에 대한 배려라고 생각했어요",
        "practical": "청결/안전을 위한 것이라고 생각했어요",
        "empathy_based": "다른 사람에게 실례가 되지 않도록 하기 위해서라고 생각했어요",
        "achievement_bias": "능숙함이나 노력을 보여주기 위한 것이라고 생각했어요",
        "symbolic": "종교적・전통적인 의미가 있다고 생각했어요",
        "indulgence_assumption": "특별하거나 편안한 자리라서 평소보다 제약이 적다고 생각했어요",
        "own_culture": "우리나라에서도 똑같이 해요",
        "no_reason": "그냥 추측 / 감으로",
        "other": "기타 (직접 입력)",
    },
    "vi": {
        "rule_based": "Tôi nghe nói đó là quy tắc cố định",
        "formality_assumption": "Tôi nghĩ đó là sự tôn trọng đối với người có địa vị cao hơn",
        "practical": "Tôi nghĩ là vì lý do sạch sẽ/an toàn",
        "empathy_based": "Để không thất lễ với người khác",
        "achievement_bias": "Tôi nghĩ đó là để thể hiện kỹ năng hoặc sự nỗ lực",
        "symbolic": "Tôi nghĩ nó mang ý nghĩa tôn giáo/truyền thống",
        "indulgence_assumption": "Tôi nghĩ sẽ có ít hạn chế hơn vì đây là dịp đặc biệt/thoải mái",
        "own_culture": "Nước tôi cũng làm giống vậy",
        "no_reason": "Chỉ đoán thôi / theo cảm tính",
        "other": "Khác (vui lòng ghi rõ)",
    },
    "id": {
        "rule_based": "Saya dengar itu aturan baku",
        "formality_assumption": "Saya pikir itu untuk menghormati orang yang berstatus lebih tinggi",
        "practical": "Saya pikir untuk kebersihan/keselamatan",
        "empathy_based": "Agar tidak tidak sopan terhadap orang lain",
        "achievement_bias": "Saya pikir itu untuk menunjukkan keterampilan atau kesungguhan",
        "symbolic": "Saya pikir ada makna agama/tradisi",
        "indulgence_assumption": "Saya pikir batasannya lebih sedikit karena ini acara spesial/santai",
        "own_culture": "Di negara saya juga melakukan hal yang sama",
        "no_reason": "Cuma menebak / mengikuti perasaan",
        "other": "Lainnya (mohon sebutkan)",
    },
}

def get_reason_options(lang_code):
    labels = REASON_LABELS.get(lang_code, REASON_LABELS["en"])
    return [(cat, labels.get(cat, REASON_LABELS["en"][cat])) for cat in REASON_CATEGORY_ORDER]

# Kept for the Research tab, which always analyzes by category (not label) and doesn't render these.
REASON_OPTIONS = get_reason_options("en")
REASON_LABEL_TO_CATEGORY = {label: cat for cat, label in REASON_OPTIONS}

# ---------- Other static quiz-screen UI strings, localized ----------
UI_STRINGS = {
    "reason_before_prompt": {
        "en": "💭 Why did you think that? (optional)",
        "ja": "💭 なぜそう思いましたか？（任意）",
        "th": "💭 ทำไมคุณถึงคิดแบบนั้น? (ไม่บังคับ)",
        "zh": "💭 你为什么这么想？（选填）",
        "ko": "💭 왜 그렇게 생각했나요? (선택 사항)",
        "vi": "💭 Tại sao bạn nghĩ vậy? (không bắt buộc)",
        "id": "💭 Mengapa Anda berpikir begitu? (opsional)",
    },
    "reason_after_prompt": {
        "en": "💡 Now that you've read the explanation, why do you think this is the custom? (optional)",
        "ja": "💡 解説を読んだ今、なぜこの習慣があると思いますか？（任意）",
        "th": "💡 หลังจากอ่านคำอธิบายแล้ว คุณคิดว่าทำไมถึงมีธรรมเนียมนี้? (ไม่บังคับ)",
        "zh": "💡 读完解说后，你认为为什么会有这个习俗？（选填）",
        "ko": "💡 설명을 읽은 지금, 왜 이런 관습이 있다고 생각하나요? (선택 사항)",
        "vi": "💡 Sau khi đọc phần giải thích, bạn nghĩ tại sao lại có phong tục này? (không bắt buộc)",
        "id": "💡 Setelah membaca penjelasannya, menurut Anda mengapa ada kebiasaan ini? (opsional)",
    },
    "please_specify": {
        "en": "Please specify:",
        "ja": "具体的に入力してください：",
        "th": "โปรดระบุ:",
        "zh": "请说明：",
        "ko": "구체적으로 입력해 주세요:",
        "vi": "Vui lòng ghi rõ:",
        "id": "Mohon jelaskan:",
    },
    "choose_correct_answer": {
        "en": "Choose the correct answer:",
        "ja": "正しい答えを選んでください：",
        "th": "เลือกคำตอบที่ถูกต้อง:",
        "zh": "请选择正确答案：",
        "ko": "정답을 선택하세요:",
        "vi": "Chọn câu trả lời đúng:",
        "id": "Pilih jawaban yang benar:",
    },
    "submit_answer": {
        "en": "Submit Answer",
        "ja": "回答する",
        "th": "ส่งคำตอบ",
        "zh": "提交答案",
        "ko": "답변 제출",
        "vi": "Gửi câu trả lời",
        "id": "Kirim Jawaban",
    },
    "select_answer_warning": {
        "en": "Please select an answer before submitting.",
        "ja": "回答する前に選択肢を選んでください。",
        "th": "กรุณาเลือกคำตอบก่อนส่ง",
        "zh": "提交前请先选择一个答案。",
        "ko": "제출하기 전에 답을 선택해 주세요.",
        "vi": "Vui lòng chọn câu trả lời trước khi gửi.",
        "id": "Silakan pilih jawaban sebelum mengirim.",
    },
    "incorrect": {
        "en": "❌ incorrect...",
        "ja": "❌ 不正解...",
        "th": "❌ ไม่ถูกต้อง...",
        "zh": "❌ 不正确……",
        "ko": "❌ 오답입니다...",
        "vi": "❌ không đúng...",
        "id": "❌ kurang tepat...",
    },
    "correct_answer_is": {
        "en": "✅ Correct answer: {answer}",
        "ja": "✅ 正解：{answer}",
        "th": "✅ คำตอบที่ถูกต้อง: {answer}",
        "zh": "✅ 正确答案：{answer}",
        "ko": "✅ 정답: {answer}",
        "vi": "✅ Đáp án đúng: {answer}",
        "id": "✅ Jawaban yang benar: {answer}",
    },
    "explanation_btn": {
        "en": "Explanation",
        "ja": "解説を見る",
        "th": "คำอธิบาย",
        "zh": "查看解说",
        "ko": "설명 보기",
        "vi": "Giải thích",
        "id": "Penjelasan",
    },
    "explanation_help": {
        "en": "Please answer 'Why did you think that?' first.",
        "ja": "先に「なぜそう思いましたか？」に答えてください。",
        "th": "กรุณาตอบ 'ทำไมคุณถึงคิดแบบนั้น?' ก่อน",
        "zh": "请先回答“你为什么这么想？”",
        "ko": "먼저 '왜 그렇게 생각했나요?'에 답해 주세요.",
        "vi": "Vui lòng trả lời 'Tại sao bạn nghĩ vậy?' trước.",
        "id": "Silakan jawab 'Mengapa Anda berpikir begitu?' terlebih dahulu.",
    },
    "next_btn": {
        "en": "Next ▶",
        "ja": "次へ ▶",
        "th": "ถัดไป ▶",
        "zh": "下一题 ▶",
        "ko": "다음 ▶",
        "vi": "Tiếp theo ▶",
        "id": "Berikutnya ▶",
    },
    "see_results_btn": {
        "en": "🏆 See Results!",
        "ja": "🏆 結果を見る！",
        "th": "🏆 ดูผลลัพธ์!",
        "zh": "🏆 查看结果！",
        "ko": "🏆 결과 보기!",
        "vi": "🏆 Xem kết quả!",
        "id": "🏆 Lihat Hasil!",
    },
    "answer_above_caption": {
        "en": "👆 Answer the question above to continue.",
        "ja": "👆 上の質問に答えると先に進めます。",
        "th": "👆 ตอบคำถามด้านบนเพื่อดำเนินการต่อ",
        "zh": "👆 请先回答上面的问题以继续。",
        "ko": "👆 위 질문에 답하면 계속 진행할 수 있어요.",
        "vi": "👆 Hãy trả lời câu hỏi phía trên để tiếp tục.",
        "id": "👆 Jawab pertanyaan di atas untuk melanjutkan.",
    },
}

def tr(key):
    lang = st.session_state.get("display_language", "en")
    table = UI_STRINGS.get(key, {})
    return table.get(lang) or table.get("en", key)

LANGUAGES = ["Abkhazian", "Afar", "Afrikaans", "Akan", "Albanian", "Amharic", "Arabic", "Aragonese", "Armenian", "Assamese", "Avaric", "Avestan", "Aymara", "Azerbaijani", "Bambara", "Bashkir", "Basque", "Belarusian", "Bengali", "Bislama", "Bosnian", "Breton", "Bulgarian", "Burmese", "Catalan", "Chamorro", "Chechen", "Chichewa", "Chinese", "Church Slavic", "Chuvash", "Cornish", "Corsican", "Cree", "Croatian", "Czech", "Danish", "Divehi", "Dutch", "Dzongkha", "English", "Esperanto", "Estonian", "Ewe", "Faroese", "Fijian", "Finnish", "French", "Fulah", "Galician", "Ganda", "Georgian", "German", "Greek", "Guarani", "Gujarati", "Haitian", "Hausa", "Hebrew", "Herero", "Hindi", "Hiri Motu", "Hungarian", "Icelandic", "Ido", "Igbo", "Indonesian", "Interlingua", "Interlingue", "Inuktitut", "Inupiaq", "Irish", "Italian", "Japanese", "Javanese", "Kalaallisut", "Kannada", "Kanuri", "Kashmiri", "Kazakh", "Khmer", "Kikuyu", "Kinyarwanda", "Kirghiz", "Komi", "Kongo", "Korean", "Kuanyama", "Kurdish", "Lao", "Latin", "Latvian", "Limburgan", "Lingala", "Lithuanian", "Luba-Katanga", "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Manx", "Maori", "Marathi", "Marshallese", "Mongolian", "Nauru", "Navajo", "Ndonga", "Nepali", "North Ndebele", "Northern Sami", "Norwegian", "Norwegian Bokmål", "Norwegian Nynorsk", "Occitan", "Ojibwa", "Oriya", "Oromo", "Ossetian", "Pali", "Panjabi", "Persian", "Polish", "Portuguese", "Pushto", "Quechua", "Romanian", "Romansh", "Rundi", "Russian", "Samoan", "Sango", "Sanskrit", "Sardinian", "Scottish Gaelic", "Serbian", "Serbo-Croatian", "Shona", "Sichuan Yi", "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "South Ndebele", "Southern Sotho", "Spanish", "Sundanese", "Swahili", "Swati", "Swedish", "Tagalog", "Tahitian", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Tibetan", "Tigrinya", "Tongan", "Tsonga", "Tswana", "Turkish", "Turkmen", "Twi", "Uighur", "Ukrainian", "Urdu", "Uzbek", "Venda", "Vietnamese", "Volapük", "Walloon", "Welsh", "Western Frisian", "Wolof", "Xhosa", "Yiddish", "Yoruba", "Zhuang", "Zulu", "Other"]

# ---------- Get available categories ----------
def load_categories():
    json_files = glob(os.path.join(DATA_DIR, "*.json"))
    return [os.path.splitext(os.path.basename(f))[0] for f in json_files]

# ---------- Hofstede cultural-dimension lookup + culture-group classification ----------
# See docs/ホフステード6次元と誤解・深化パターン仮説.md for the methodology and per-dimension
# hypotheses behind this. Scores come from data/country_culture_table.csv, which maps every
# nationality in COUNTRIES to either a direct Hofstede score, a proxy country/region score, or
# "unknown" when no sufficiently confident proxy exists.
HOFSTEDE_DIMS = ["pdi", "idv", "mas", "uai", "ltowvs", "ivr"]
CULTURE_TABLE_FILE = os.path.join(DATA_DIR, "country_culture_table.csv")

# Representative countries for each hypothesized culture group (from the grouping-hypothesis doc).
# Group centroids are computed as the mean score per dimension across whichever representative
# countries actually have a value for that dimension (direct or proxy).
CULTURE_GROUP_COUNTRIES = {
    "A. Western (US/UK/Northwest Europe)": [
        "United States", "United Kingdom", "Germany", "Netherlands", "Australia",
        "Sweden", "Norway", "Denmark", "Finland", "Canada",
    ],
    "B. Southern Europe": [
        "France", "Italy", "Spain", "Portugal",
    ],
    "C. East Asia": [
        "China", "Korea, Republic of", "Taiwan, Province of China", "Hong Kong",
    ],
    "D. Southeast / South Asia": [
        "Thailand", "Indonesia", "Philippines", "India", "Malaysia",
    ],
    "E. Middle East / Arab world": [
        "United Arab Emirates", "Saudi Arabia", "Egypt", "Lebanon",
    ],
    "F. Latin America": [
        "Mexico", "Brazil", "Argentina", "Chile", "Colombia",
    ],
}

@st.cache_data
def load_culture_table():
    """nationality (as in COUNTRIES) -> dict of status/confidence/6 Hofstede dimension scores."""
    table = {}
    if not os.path.exists(CULTURE_TABLE_FILE):
        return table
    df = read_csv(CULTURE_TABLE_FILE)
    for _, row in df.iterrows():
        scores = {}
        for dim in HOFSTEDE_DIMS:
            val = row.get(dim)
            scores[dim] = None if val is None or (isinstance(val, float) and val != val) else float(val)
        table[row["country"]] = {
            "status": row.get("status"),
            "confidence": row.get("confidence"),
            "scores": scores,
        }
    return table

def get_hofstede_scores(nationality):
    """Return a dict of the 6 Hofstede dimension scores for a nationality, or None if unknown."""
    table = load_culture_table()
    entry = table.get(nationality)
    if not entry or entry["status"] == "unknown":
        return None
    return entry["scores"]

@st.cache_data
def compute_culture_group_centroids():
    """group name -> {dim: mean score across representative countries that have that dim}"""
    table = load_culture_table()
    centroids = {}
    for group, countries in CULTURE_GROUP_COUNTRIES.items():
        sums = {d: [] for d in HOFSTEDE_DIMS}
        for c in countries:
            entry = table.get(c)
            if not entry:
                continue
            for d in HOFSTEDE_DIMS:
                v = entry["scores"].get(d)
                if v is not None:
                    sums[d].append(v)
        centroids[group] = {d: (sum(vals) / len(vals) if vals else None) for d, vals in sums.items()}
    return centroids

@st.cache_data
def compute_dimension_scale():
    """Per-dimension standard deviation across every country with a value, used to normalize
    each Hofstede dimension before computing distances. Without this, dimensions with a larger
    raw spread (e.g. UAI) would dominate the nearest-centroid classification."""
    table = load_culture_table()
    scale = {}
    for d in HOFSTEDE_DIMS:
        vals = [e["scores"][d] for e in table.values() if e["scores"].get(d) is not None]
        if len(vals) < 2:
            scale[d] = 1.0
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        scale[d] = variance ** 0.5 or 1.0
    return scale

def get_culture_group(nationality):
    """Classify a nationality into the nearest culture group by (normalized) Euclidean distance
    over whichever Hofstede dimensions both the country and the group centroid have a value for.
    Returns None if the country has no usable score at all (status == 'unknown')."""
    scores = get_hofstede_scores(nationality)
    if not scores:
        return None
    centroids = compute_culture_group_centroids()
    scale = compute_dimension_scale()
    best_group, best_dist = None, None
    for group, centroid in centroids.items():
        diffs = []
        for d in HOFSTEDE_DIMS:
            a, b = scores.get(d), centroid.get(d)
            if a is not None and b is not None:
                diffs.append(((a - b) / scale[d]) ** 2)
        if not diffs:
            continue
        dist = sum(diffs) ** 0.5
        if best_dist is None or dist < best_dist:
            best_group, best_dist = group, dist
    return best_group

# ---------- User management ----------
USERS_FILE    = os.path.join(os.path.dirname(__file__), "results", "users.json")
AUTH_LOG_FILE = os.path.join(os.path.dirname(__file__), "results", "auth_log.csv")
SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "results", "sessions.json")

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def log_auth(event, username, success, note=""):
    """event: 'login' or 'register'"""
    os.makedirs(os.path.dirname(AUTH_LOG_FILE), exist_ok=True)
    write_header = not os.path.exists(AUTH_LOG_FILE) or os.path.getsize(AUTH_LOG_FILE) == 0
    with open(AUTH_LOG_FILE, "a", encoding="utf-8") as f:
        if write_header:
            f.write("datetime,event,username,success,note\n")
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},{event},{username},{success},{note}\n')

def normalize_user_record(v):
    """Normalize into a dict with password/nationality/native_language"""
    if v is None:
        return {"password": None, "nationality": "", "native_language": ""}
    if isinstance(v, str):
        return {"password": v, "nationality": "", "native_language": ""}
    if isinstance(v, dict):
        v.setdefault("password", None)
        v.setdefault("nationality", "")
        v.setdefault("native_language", "")
        return v
    return {"password": None, "nationality": "", "native_language": ""}

def load_users():
    if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
        with open(USERS_FILE) as f:
            data = json.load(f)
        # Migrate old format (list) to new format (dict)
        if isinstance(data, list):
            data = {name: None for name in data}
        changed = False
        for name, v in list(data.items()):
            normalized = normalize_user_record(v)
            if normalized != v:
                changed = True
            data[name] = normalized
        if changed:
            with open(USERS_FILE, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        return data
    return {}

def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, ensure_ascii=False)

def get_user_profile(username):
    """Helper to fetch user attributes, used when saving quiz_results, etc."""
    users = load_users()
    record = users.get(username) or {}
    return {
        "nationality": record.get("nationality", ""),
        "native_language": record.get("native_language", ""),
    }

def load_sessions():
    if os.path.exists(SESSIONS_FILE) and os.path.getsize(SESSIONS_FILE) > 0:
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, ensure_ascii=False)

def create_session(username):
    """Generate a token, set it in the URL, and save it to file"""
    token = secrets.token_urlsafe(32)
    sessions = load_sessions()
    sessions[token] = username
    save_sessions(sessions)
    st.query_params["session"] = token
    return token

def delete_session():
    """Remove the token on logout"""
    token = st.query_params.get("session")
    if token:
        sessions = load_sessions()
        sessions.pop(token, None)
        save_sessions(sessions)
        st.query_params.clear()

# ---------- Restore session on reload ----------
if "username" not in st.session_state:
    token = st.query_params.get("session")
    if token:
        sessions = load_sessions()
        if token in sessions:
            st.session_state.username = sessions[token]

# ---------- Login page ----------
if "username" not in st.session_state:
    st.title("🗾 Let's Learn Japanese Manners!")

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    # ---- Login tab ----
    with tab_login:
        st.markdown("Welcome back! Please log in.")
        login_name = st.text_input("Username:", key="login_name")
        login_pw   = st.text_input("Password:", type="password", key="login_pw")
        if st.button("Login", use_container_width=True):
            name = login_name.strip()
            pw   = login_pw.strip()
            if not name or not pw:
                st.warning("Please enter both username and password.")
            else:
                users = load_users()
                if name not in users:
                    log_auth("login", name, False, "username not found")
                    st.error(f"❌ '{name}' is not registered yet.")
                    st.info("👉 Go to the **Register** tab above to create a new account!")
                elif users[name]["password"] is None:
                    # Legacy data (no password set) -> auto-set password and log in
                    users[name]["password"] = hash_pw(pw)
                    save_users(users)
                    log_auth("login", name, True, "legacy account - password set")
                    st.session_state.username = name
                    create_session(name)
                    st.rerun()
                elif users[name]["password"] == hash_pw(pw):
                    log_auth("login", name, True, "")
                    st.session_state.username = name
                    create_session(name)
                    st.rerun()
                else:
                    log_auth("login", name, False, "wrong password")
                    st.error("❌ Wrong password. Please try again.")

    # ---- Register tab ----
    with tab_register:
        st.markdown("Create a new account.")

        with st.expander("📄 Research Participation Terms & Conditions / 研究参加への同意事項", expanded=True):
            st.markdown("""
**About this study / この研究について**

Mannavi is a research tool built for an undergraduate thesis project, studying how foreign
visitors' understanding of Japanese manners deepens through this quiz, and how that process
differs by cultural background.
（本アプリは卒業論文研究の一環として作成されたツールです。外国人観光客が日本のマナーに
関する理解をどのように深めるか、また出身文化圏によってその過程がどう異なるかを研究しています。）

**What participation involves / 参加して行うこと**

You will answer short multiple-choice quiz questions about Japanese manners, and for each
one, select a reason for your answer before and after reading an explanation. You will also
provide your nationality and native language.
（日本のマナーに関する短いクイズに回答し、各設問について解説を読む前後で「なぜそう考えたか」という
理由を選択します。国籍と母国語も入力していただきます。）

**Data use / データの利用について**

Your answers, selected reasons, nationality, and native language are recorded and used only
for academic analysis in this thesis research. Your username is not your real name and is not
published; results will be reported in aggregate or anonymized form.
（回答内容・選択した理由・国籍・母国語は、この卒業論文研究の学術的な分析にのみ使用されます。
ユーザー名は実名である必要はなく、公開されることもありません。結果は集計・匿名化した形で報告されます。）

If you have any questions, please contact the researcher (Riko Homma / gmail: a23.mxxw@g.chuo-u.ac.jp) directly.
            """)
            agree = st.checkbox(
                "I have read and understood the above, and I agree to participate in this "
                "research. / 私は上記の内容を読み理解し、本研究への参加に同意します。",
                key="consent_agree",
            )

        if not agree:
            st.info("Please check the agreement box above to continue with registration. / "
                    "登録を続けるには、上の同意チェックボックスにチェックを入れてください。")
        else:
            reg_name  = st.text_input("Username:", key="reg_name")
            reg_pw    = st.text_input("Password:", type="password", key="reg_pw")
            reg_pw2   = st.text_input("Confirm Password:", type="password", key="reg_pw2")

            nationality = st.selectbox(
                "Nationality / Country you're from:",
                options=COUNTRIES,
                index=None,
                placeholder="Type to search (e.g. 'Ja' → Japan)...",
                key="reg_nationality",
            )
            if nationality == "Other":
                nationality = st.text_input("Please specify your nationality:", key="reg_nationality_other").strip()

            native_lang = st.selectbox(
                "Native language:",
                options=LANGUAGES,
                index=None,
                placeholder="Type to search (e.g. 'Jap' → Japanese)...",
                key="reg_native_lang",
            )
            if native_lang == "Other":
                native_lang = st.text_input("Please specify your native language:", key="reg_native_lang_other").strip()

            if st.button("Register", use_container_width=True):
                name = reg_name.strip()
                pw   = reg_pw.strip()
                pw2  = reg_pw2.strip()
                nationality = (nationality or "").strip()
                native_lang = (native_lang or "").strip()
                if not name or not pw or not pw2 or not nationality or not native_lang:
                    st.warning("Please fill in all fields.")
                elif pw != pw2:
                    st.error("❌ Passwords do not match.")
                else:
                    with file_lock(USERS_FILE):
                        users = load_users()
                        if name in users and users[name].get("password") is not None:
                            log_auth("register", name, False, "username already taken")
                            st.error(f"❌ '{name}' is already taken. Please choose a different username.")
                        else:
                            users[name] = {
                                "password": hash_pw(pw),
                                "nationality": nationality,
                                "native_language": native_lang,
                                "consented_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            save_users(users)
                            log_auth("register", name, True, "")
                            st.session_state.username = name
                            create_session(name)
                            st.rerun()

    st.stop()

# Usernames that get access to the "Research" tab (cross-user culture-group analysis).
# Add your own account's username here.
RESEARCHER_USERNAMES = {"Riko"}

CATEGORY_META = {
    "communication":    {"icon": "💬", "label": "Communication"},
    "onsen":            {"icon": "♨️",  "label": "Onsen"},
    "public_transport": {"icon": "🚃", "label": "Public Transport"},
    "questions":        {"icon": "🙏", "label": "General Manners"},
    "shrine":           {"icon": "⛩️",  "label": "Shrine"},
    "dining":           {"icon": "🍜", "label": "Dining & Izakaya"},
    "workplace":        {"icon": "🏢", "label": "Workplace"},
    "gifts":            {"icon": "🎁", "label": "Gift-Giving"},
}

# ---------- Display-language translation overlay ----------
# Pre-translated copies of the quiz content (question/options/hint/explanation only)
# live under data/translations/<lang_code>/<category>.json, mirroring the English files.
TRANSLATIONS_DIR = os.path.join(DATA_DIR, "translations")

# Maps a profile's "native_language" value to the translation folder to default to.
NATIVE_LANGUAGE_TO_CODE = {
    "Thai": "th",
    "Chinese": "zh",
    "Korean": "ko",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Japanese": "ja",
}

DISPLAY_LANGUAGES = {
    "en": "🇬🇧 English",
    "ja": "🇯🇵 日本語 (Japanese)",
    "th": "🇹🇭 ไทย (Thai)",
    "zh": "🇨🇳 中文 (Chinese)",
    "ko": "🇰🇷 한국어 (Korean)",
    "vi": "🇻🇳 Tiếng Việt (Vietnamese)",
    "id": "🇮🇩 Bahasa Indonesia (Indonesian)",
}

@st.cache_data
def load_translation(category, lang_code):
    if lang_code == "en":
        return None
    path = os.path.join(TRANSLATIONS_DIR, lang_code, f"{category}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def localize_questions(category, questions_en, lang_code):
    """Overlay translated question/options/hint/explanation onto the English question
    data, leaving answer/option_tags/image/category untouched. Falls back silently to
    English if no translation file exists yet, or its item count doesn't line up."""
    translated = load_translation(category, lang_code)
    if not translated or len(translated) != len(questions_en):
        return questions_en
    merged = []
    for orig, tr in zip(questions_en, translated):
        item = dict(orig)
        for key in ("question", "options", "hint", "explanation"):
            if key in tr:
                item[key] = tr[key]
        merged.append(item)
    return merged

def start_quiz(selected):
    file_path = os.path.join(DATA_DIR, f"{selected}.json")
    with open(file_path, "r") as f:
        selected_questions = json.load(f)
    lang_code = st.session_state.get("display_language", "en")
    selected_questions = localize_questions(selected, selected_questions, lang_code)
    indices = list(range(len(selected_questions)))
    random.shuffle(indices)
    indices = indices[:4]
    st.session_state.category = selected
    st.session_state.questions = selected_questions
    st.session_state.question_order = indices
    st.session_state.question_index = 0
    st.session_state.answered = False
    st.session_state.correct = None
    st.session_state.consecutive_correct = 0
    st.session_state.show_explanation = False
    st.session_state.show_celebration = False
    st.session_state.celebration = None
    st.session_state.correct_list = []
    st.session_state.answer_log = []

# ---------- Home page ----------
if "category" not in st.session_state:
    # ---- Title row and logout button ----
    col_title, col_logout = st.columns([8, 1])
    col_title.title("🗾 Let's Learn Japanese Manners!")
    col_title.markdown(f"Welcome, **{st.session_state.username}**!")

    with col_logout:
        st.markdown("""
        <style>
        div[data-testid="stButton"].logout-wrap > button {
            height: 2rem !important;
            min-height: unset !important;
            padding: 0 0.75rem !important;
            font-size: 0.8rem !important;
            line-height: 2rem !important;
            white-space: nowrap !important;
        }
        </style>
        <div class="logout-wrap">
        """, unsafe_allow_html=True)
        clicked = st.button("Log out", use_container_width=True, key="logout_btn")
        st.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state.confirm_logout = True

    # ---- Logout confirmation dialog ----
    if st.session_state.get("confirm_logout"):
        st.warning("Are you sure you want to log out?")
        c_yes, c_no = st.columns(2)
        if c_yes.button("✅ Yes", use_container_width=True):
            delete_session()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        if c_no.button("❌ No", use_container_width=True):
            st.session_state.confirm_logout = False
            st.rerun()
        st.stop()

    if "display_language" not in st.session_state:
        profile = get_user_profile(st.session_state.username)
        st.session_state.display_language = NATIVE_LANGUAGE_TO_CODE.get(profile.get("native_language"), "en")

    is_researcher = st.session_state.username in RESEARCHER_USERNAMES
    if is_researcher:
        tab_quiz, tab_portfolio, tab_research = st.tabs(["🎯 Quiz", "📊 My Portfolio", "🔬 Research"])
    else:
        tab_quiz, tab_portfolio = st.tabs(["🎯 Quiz", "📊 My Portfolio"])

    # -------------------- Portfolio tab --------------------
    with tab_portfolio:
        with st.expander("⚙️ Profile settings / プロフィール設定（国籍・母国語を後から変更）"):
            profile = get_user_profile(st.session_state.username)
            current_nat = profile.get("nationality") or None
            current_lang = profile.get("native_language") or None

            new_nationality = st.selectbox(
                "Nationality / 出身国:",
                options=COUNTRIES,
                index=COUNTRIES.index(current_nat) if current_nat in COUNTRIES else None,
                placeholder="Type to search (e.g. 'Ja' → Japan)...",
                key="profile_edit_nationality",
            )
            if new_nationality == "Other":
                new_nationality = st.text_input(
                    "Please specify your nationality / 具体的に入力してください:",
                    value=current_nat if current_nat not in COUNTRIES else "",
                    key="profile_edit_nationality_other",
                ).strip()

            new_native_lang = st.selectbox(
                "Native language / 母国語:",
                options=LANGUAGES,
                index=LANGUAGES.index(current_lang) if current_lang in LANGUAGES else None,
                placeholder="Type to search (e.g. 'Jap' → Japanese)...",
                key="profile_edit_native_lang",
            )
            if new_native_lang == "Other":
                new_native_lang = st.text_input(
                    "Please specify your native language / 具体的に入力してください:",
                    value=current_lang if current_lang not in LANGUAGES else "",
                    key="profile_edit_native_lang_other",
                ).strip()

            if st.button("Save profile / 保存する", key="profile_edit_save"):
                new_nationality = (new_nationality or "").strip()
                new_native_lang = (new_native_lang or "").strip()
                if not new_nationality or not new_native_lang:
                    st.warning("Please fill in both fields. / 両方とも入力してください。")
                else:
                    with file_lock(USERS_FILE):
                        users = load_users()
                        record = users.get(st.session_state.username) or {}
                        record["nationality"] = new_nationality
                        record["native_language"] = new_native_lang
                        users[st.session_state.username] = record
                        save_users(users)
                    st.success("Saved! / 保存しました。")
                    st.rerun()

        RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "quiz_results.csv")
        if not os.path.exists(RESULTS_PATH) or os.path.getsize(RESULTS_PATH) == 0:
            st.info("No results yet. Complete a quiz to see your portfolio!")
        else:
            import pandas as pd
            import datetime as dt
            df_all = pd.read_csv(RESULTS_PATH)
            df_all = df_all[df_all["username"] == st.session_state.username].copy()
            df_all = df_all[df_all["q_num"].notna()].copy()

            if df_all.empty:
                st.info("No results yet. Complete a quiz to see your portfolio!")
            else:
                df_all["datetime"] = pd.to_datetime(df_all["datetime"])
                df_all["is_correct"] = df_all["is_correct"].astype(str).str.lower() == "true"

                # ---------- Summary ----------
                total_q   = len(df_all)
                total_cor = df_all["is_correct"].sum()
                accuracy  = total_cor / total_q * 100
                sessions  = df_all.groupby("datetime").ngroups

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Questions", total_q)
                c2.metric("Correct", int(total_cor))
                c3.metric("Accuracy", f"{accuracy:.1f}%")
                c4.metric("Sessions", sessions)

                st.divider()

                # ---------- Accuracy by category (bar chart) ----------
                st.subheader("📂 Accuracy by Category")
                cat_stats = (
                    df_all.groupby("category")["is_correct"]
                    .agg(correct="sum", total="count")
                    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
                    .reset_index()
                )
                cat_stats["label"] = cat_stats["category"].map(
                    lambda c: f"{CATEGORY_META.get(c, {}).get('icon','📚')} {CATEGORY_META.get(c, {}).get('label', c)}"
                )
                st.bar_chart(cat_stats.set_index("label")["accuracy"], y_label="Accuracy (%)", height=280)

                st.divider()

                # ---------- Score trend by session (line chart) ----------
                st.subheader("📈 Score Over Time")
                session_scores = (
                    df_all.groupby("datetime")["is_correct"]
                    .agg(correct="sum", total="count")
                    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
                    .reset_index()
                    .sort_values("datetime")
                )
                session_scores["date"] = session_scores["datetime"].dt.strftime("%m/%d %H:%M")
                st.line_chart(session_scores.set_index("date")["accuracy"], y_label="Accuracy (%)", height=250)

                st.divider()

                # ---------- Weak questions list ----------
                st.subheader("❌ Questions to Review")
                weak = (
                    df_all.groupby("question")["is_correct"]
                    .agg(correct="sum", total="count")
                    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
                    .reset_index()
                    .query("accuracy < 100")
                    .sort_values("accuracy")
                )
                if weak.empty:
                    st.success("Perfect! No weak points found.")
                else:
                    for _, row in weak.iterrows():
                        emoji = "🔴" if row["accuracy"] == 0 else "🟡"
                        st.markdown(
                            f"{emoji} **{row['question']}**  "
                            f"&nbsp;&nbsp;{int(row['correct'])}/{int(row['total'])} correct "
                            f"({row['accuracy']:.0f}%)"
                        )

    # -------------------- Research tab (researcher-only, all users) --------------------
    if is_researcher:
        with tab_research:
            st.caption(
                "研究用の問い①（規則ベース→共感ベースへの深化）と問い②（文化圏ごとの誤解パターン）を、"
                "全利用者のデータを横断して分析します。手法の詳細はdocs/ホフステード6次元と誤解・深化パターン仮説.mdを参照してください。"
            )
            if st.button("🔄 最新のデータに更新"):
                st.rerun()
            st.caption("※このページは自動更新されません。新しい回答を反映したいときはこのボタンを押してください（ブラウザの再読み込みでも同じです）。")
            RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "quiz_results.csv")
            if not os.path.exists(RESULTS_PATH) or os.path.getsize(RESULTS_PATH) == 0:
                st.info("まだ結果がありません。")
            else:
                with open(RESULTS_PATH, "rb") as f:
                    st.download_button(
                        "📥 quiz_results.csv をダウンロード",
                        data=f.read(),
                        file_name="quiz_results.csv",
                        mime="text/csv",
                        help="Streamlit Community Cloudの保存領域は一時的です。アプリが再起動するとデータが消えることがあるので、定期的にダウンロードしてください。",
                    )
                df = read_csv(RESULTS_PATH)
                # Older rows saved before these columns existed won't have them;
                # add them as all-NaN so the analysis below doesn't KeyError.
                for col in ["selected_tag", "reason_category_before", "reason_text_before",
                            "reason_category_after", "reason_text_after", "nationality"]:
                    if col not in df.columns:
                        df[col] = None
                df = df[df["q_num"].notna()].copy()
                if df.empty:
                    st.info("まだ結果がありません。")
                else:
                    df["culture_group"] = df["nationality"].apply(get_culture_group)
                    n_unclassified = df["culture_group"].isna().sum()

                    st.subheader("🌍 文化圏別の回答者数")
                    group_counts = df.groupby("culture_group", dropna=False).size().reset_index(name="rows")
                    group_counts["culture_group"] = group_counts["culture_group"].fillna("（国籍不明／未対応）")
                    st.bar_chart(group_counts.set_index("culture_group")["rows"], height=250)
                    st.caption(f"{n_unclassified}件の回答は、国籍からホフステードの信頼できるスコアが得られないため、以下の集計からは除外されています。")

                    df_known = df[df["culture_group"].notna()].copy()

                    st.divider()

                    # ---------- RQ2: culture group x misconception tag ----------
                    st.subheader("② 文化圏 × 誤答タグ（selected_tag）")
                    if df_known["selected_tag"].notna().any():
                        tag_tab = (
                            df_known[df_known["selected_tag"].notna()]
                            .groupby(["culture_group", "selected_tag"])
                            .size()
                            .unstack(fill_value=0)
                        )
                        tag_tab_pct = tag_tab.div(tag_tab.sum(axis=1), axis=0) * 100
                        st.dataframe(tag_tab_pct.round(1).astype(str) + "%")
                        st.caption("行単位で正規化：その文化圏の回答のうち、各誤答タグが占める割合（%）。")
                    else:
                        st.info("タグ付きの回答がまだありません（selected_tagは誤答・distractorの場合のみ記録されます）。")

                    st.divider()

                    # ---------- RQ1: before -> after reason-category shift ----------
                    st.subheader("① 解説を読んだ後の理由カテゴリの変化")
                    # formality_assumption/achievement_bias/indulgence_assumption are, like
                    # own_culture, projected general assumptions rather than reasoning engaged
                    # with this specific custom -> grouped with SURFACE (see REASON_CATEGORY_ORDER
                    # above for the Hofstede-dimension mapping this mirrors).
                    SURFACE = {"rule_based", "no_reason", "own_culture", "formality_assumption", "achievement_bias", "indulgence_assumption"}
                    REASONED = {"practical", "symbolic", "empathy_based"}

                    def bucket(cat):
                        if cat in SURFACE:
                            return "外的な規則・借用的理由"
                        if cat in REASONED:
                            return "自分なりの理由づけ"
                        return None

                    # Overall shift rate only needs the before/after reason pair, not a known
                    # culture group — so it's computed from the full dataframe (df), not df_known.
                    # Rows with no nationality on file (e.g. a profile that never set one) would
                    # otherwise be silently dropped from this headline number.
                    shift_df_all = df[df["reason_category_before"].notna() & df["reason_category_after"].notna()].copy()
                    if shift_df_all.empty:
                        st.info("解答前後の理由カテゴリのペアがまだ記録されていません。")
                    else:
                        shift_df_all["bucket_before"] = shift_df_all["reason_category_before"].apply(bucket)
                        shift_df_all["bucket_after"] = shift_df_all["reason_category_after"].apply(bucket)
                        shift_df_all["deepened"] = (shift_df_all["bucket_before"] == "外的な規則・借用的理由") & (shift_df_all["bucket_after"] == "自分なりの理由づけ")

                        overall_rate = shift_df_all["deepened"].mean() * 100
                        st.metric("全体：外的な規則 → 自分なりの理由づけ への移行率", f"{overall_rate:.1f}%")
                        n_no_group = shift_df_all["culture_group"].isna().sum()
                        if n_no_group:
                            st.caption(f"全体の件数には、国籍が未設定・未対応で文化圏が分からない{n_no_group}件も含みます。下の文化圏別の表には含まれません。")

                        # Per-culture-group breakdown still needs a known culture group.
                        shift_df = shift_df_all[shift_df_all["culture_group"].notna()]
                        if not shift_df.empty:
                            by_group = (
                                shift_df.groupby("culture_group")["deepened"]
                                .agg(shift_rate=lambda x: x.mean() * 100, n="count")
                                .round(1)
                            )
                            st.dataframe(by_group)

                            st.caption("文化圏別の理由カテゴリ内訳（解説を読んだ後）：")
                            after_tab = (
                                shift_df.groupby(["culture_group", "reason_category_after"])
                                .size()
                                .unstack(fill_value=0)
                            )
                            after_tab_pct = after_tab.div(after_tab.sum(axis=1), axis=0) * 100
                            st.dataframe(after_tab_pct.round(1).astype(str) + "%")
                        else:
                            st.caption("文化圏別の内訳は、国籍から文化圏が判定できるデータが集まるまで表示されません。")

    # -------------------- Quiz tab --------------------
    with tab_quiz:
        lang_codes = list(DISPLAY_LANGUAGES.keys())
        current_lang = st.session_state.get("display_language", "en")
        st.session_state.display_language = st.selectbox(
            "🌐 Quiz language / 表示言語",
            options=lang_codes,
            index=lang_codes.index(current_lang) if current_lang in lang_codes else 0,
            format_func=lambda code: DISPLAY_LANGUAGES[code],
            help="Questions, hints and explanations will show in this language where a translation is available.",
        )

        st.markdown("Choose a situation:")

        st.markdown("""
    <style>
    div[data-testid="stButton"].cat-btn > button {
        height: 130px;
        width: 100%;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 16px;
        white-space: pre-line;
    }
    </style>
    """, unsafe_allow_html=True)

        # During countdown
        if st.session_state.get("countdown_category"):
            cat = st.session_state.countdown_category
            meta = CATEGORY_META.get(cat, {"icon": "📚", "label": cat.replace("_", " ").title()})
            placeholder = st.empty()
            steps = [
                ("3", "#e74c3c", "7rem"),
                ("2", "#e67e22", "7rem"),
                ("1", "#27ae60", "7rem"),
                (f"{meta['icon']}<br>START!", "#1a73e8", "4.5rem"),
            ]
            durations = [0.75, 0.75, 0.75, 1.5]
            for (text, color, size), duration in zip(steps, durations):
                placeholder.markdown(
                    f"<div style='text-align:center;font-size:{size};font-weight:900;"
                    f"color:{color};padding:60px 0;'>{text}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(duration)
            placeholder.empty()
            start_quiz(cat)
            del st.session_state.countdown_category
            st.rerun()
            st.stop()

        # Required-first ordering for the experiment: the 6 core categories (covering all
        # 6 Hofstede dimensions) come first, then the 2 optional/bonus ones.
        CATEGORY_PRIORITY = ["dining", "workplace", "gifts", "onsen", "shrine", "communication",
                             "public_transport", "questions"]
        categories = sorted(
            load_categories(),
            key=lambda c: CATEGORY_PRIORITY.index(c) if c in CATEGORY_PRIORITY else len(CATEGORY_PRIORITY)
        )
        cols = st.columns(3)
        for i, cat in enumerate(categories):
            meta = CATEGORY_META.get(cat, {"icon": "📚", "label": cat.replace("_", " ").title()})
            label = f"{meta['icon']}\n{meta['label']}"
            with cols[i % 3]:
                st.markdown('<div class="cat-btn">', unsafe_allow_html=True)
                if st.button(label, key=f"cat_{cat}", use_container_width=True):
                    st.session_state.countdown_category = cat
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ---------- Get state variables ----------
if "answer_log" not in st.session_state:
    st.session_state.answer_log = []

questions = st.session_state.questions
order = st.session_state.question_order
current_idx = st.session_state.question_index

# ---------- Finish screen ----------
if current_idx >= len(order):
    total = len(st.session_state.correct_list)
    corrects = sum(st.session_state.correct_list)

    # Auto-save (once only)
    if not st.session_state.get("score_saved"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "quiz_results.csv")

        profile = get_user_profile(st.session_state.username)

        rows = []
        for log in st.session_state.answer_log:
            rows.append({
                "datetime": now,
                "username": st.session_state.username,
                "nationality": profile["nationality"],
                "native_language": profile["native_language"],
                "category": st.session_state.category,
                "q_num": log["q_num"],
                "question": log["question"],
                "user_answer": log["user_answer"],
                "correct_answer": log["correct_answer"],
                "is_correct": log["is_correct"],
                "selected_tag": log.get("selected_tag"),
                "reason_category_before": log.get("reason_category_before"),
                "reason_text_before": log.get("reason_text_before", ""),
                "reason_category_after": log.get("reason_category_after"),
                "reason_text_after": log.get("reason_text_after", ""),
                "score": f"{corrects} out of {total}",
            })
        df = DataFrame(rows)
        with file_lock(path):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                df_old = read_csv(path)
                df_write = concat([df_old, df], ignore_index=True)
            else:
                df_write = df
            df_write.to_csv(path, index=False)
        st.session_state.score_saved = True

    st.title("🎉 You Finished All Questions!")
    st.markdown(f"**Final Score: {corrects} / {total} correct**")
    st.progress(corrects / total)
    st.caption("📥 Score automatically saved to CSV.")

    st.markdown("### Results:")
    for log in st.session_state.answer_log:
        emoji = "✅" if log["is_correct"] else "❌"
        st.write(f"Q{log['q_num']}: {emoji} {log['question']}")
        if not log["is_correct"]:
            st.caption(f"　Your answer: {log['user_answer']}  →  Correct: {log['correct_answer']}")

    if st.button("Restart with different category"):
        username = st.session_state.username
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.username = username
        st.rerun()
    st.stop()

# ---------- Current question ----------
q = questions[order[current_idx]]

col_title, col_exit = st.columns([5, 1])
col_title.subheader(f"Question {current_idx + 1} : {st.session_state.category.title()}")
if col_exit.button("🚪 Exit", help="Back to home"):
    username = st.session_state.username
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.username = username
    st.rerun()

st.caption(f"🔥 {st.session_state.consecutive_correct} consecutive correct  •  Question {current_idx + 1} / {len(order)}")
st.progress((current_idx + 1) / len(order))

st.markdown(f"##### {q['question']}")
col_l, col_c, col_r = st.columns([1, 1, 1])
with col_c:
    st.image(resolve_image_path(q["image"]), width=140)

user_answer = st.radio(tr("choose_correct_answer"), q["options"], index=None)

# ---------- Submit ----------
CELEBRATIONS = [
    {
        "id": "balloons",
        "fx": "balloons",
        "html": "<div style='text-align:center;font-size:2.8rem;font-weight:bold;color:#28a745;animation:none'>🎈 CORRECT!!! 🎈<br><span style='font-size:1.4rem;color:#555'>Nice one!</span></div>",
    },
    {
        "id": "snow",
        "fx": "snow",
        "html": "<div style='text-align:center;font-size:2.8rem;font-weight:bold;color:#1a73e8'>❄️ BRILLIANT!!! ❄️<br><span style='font-size:1.4rem;color:#555'>Keep it up!</span></div>",
    },
    {
        "id": "rainbow",
        "fx": "balloons",
        "html": "<div style='text-align:center;font-size:2.8rem;font-weight:bold;background:linear-gradient(90deg,#ff0000,#ff9900,#33cc33,#0099ff,#9900cc);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>🌈 AMAZING!!! 🌈<br></div><div style='text-align:center;font-size:1.4rem;color:#555'>Fantastic answer!</div>",
    },
    {
        "id": "star",
        "fx": "balloons",
        "html": "<div style='text-align:center;font-size:2.8rem;font-weight:bold;color:#f0a500'>⭐ EXCELLENT!!! ⭐<br><span style='font-size:1.4rem;color:#555'>You're shining!</span></div>",
    },
    {
        "id": "rocket",
        "fx": "snow",
        "html": "<div style='text-align:center;font-size:2.8rem;font-weight:bold;color:#9900cc'>🚀 OUT OF THIS WORLD!!! 🚀<br><span style='font-size:1.4rem;color:#555'>Superb!</span></div>",
    },
]

STREAK_CELEBRATION = {
    "fx": "balloons",
    "html": "<div style='text-align:center;font-size:3rem;font-weight:bold;color:#ff6b00'>🔥 {n} STREAK! 🔥<br>UNSTOPPABLE!!!</div>",
}

if st.button(tr("submit_answer"), key=f"submit_answer_{current_idx}") and not st.session_state.answered:
    if user_answer is None:
        st.warning(tr("select_answer_warning"))
        st.stop()
    st.session_state.answered = True
    is_correct = (user_answer == q["options"][q["answer"]])
    st.session_state.correct = is_correct
    st.session_state.correct_list.append(is_correct)
    selected_idx = q["options"].index(user_answer)
    selected_tag = q.get("option_tags", [None] * len(q["options"]))[selected_idx]
    st.session_state.answer_log.append({
        "q_num": current_idx + 1,
        "question": q["question"],
        "user_answer": user_answer,
        "correct_answer": q["options"][q["answer"]],
        "is_correct": is_correct,
        "selected_tag": selected_tag,
        "reason_category_before": None,
        "reason_text_before": "",
        "reason_category_after": None,
        "reason_text_after": "",
    })
    if is_correct:
        st.session_state.consecutive_correct += 1
        st.session_state.celebration = random.choice(CELEBRATIONS)
        st.session_state.show_celebration = True
    else:
        st.session_state.consecutive_correct = 0
        st.session_state.show_celebration = False

# ---------- After answering ----------
if st.session_state.answered:
    if st.session_state.correct:
        consecutive = st.session_state.consecutive_correct
        if consecutive >= 3:
            st.markdown(
                STREAK_CELEBRATION["html"].format(n=consecutive),
                unsafe_allow_html=True,
            )
        else:
            cel = st.session_state.get("celebration", CELEBRATIONS[0])
            st.markdown(cel["html"], unsafe_allow_html=True)
    else:
        st.error(tr("incorrect"))
        st.info(tr("correct_answer_is").format(answer=q["options"][q["answer"]]))

    lang = st.session_state.get("display_language", "en")
    reason_options_loc = get_reason_options(lang)
    reason_label_to_category_loc = {label: cat for cat, label in reason_options_loc}
    reason_labels = [label for _, label in reason_options_loc]

    # ---- Reason, captured BEFORE the explanation is revealed ----
    reason_choice_before = st.radio(
        tr("reason_before_prompt"),
        reason_labels,
        index=None,
        key=f"reason_choice_{current_idx}",
    )
    reason_category_before = None
    reason_text_before = ""
    if reason_choice_before:
        reason_category_before = reason_label_to_category_loc[reason_choice_before]
        if reason_category_before == "other":
            reason_text_before = st.text_input(
                tr("please_specify"),
                key=f"reason_other_{current_idx}",
            ).strip()
        else:
            reason_text_before = reason_choice_before

    # ---- Reason, captured AFTER the explanation is revealed (only shown once they've viewed it) ----
    reason_category_after = None
    reason_text_after = ""
    if st.session_state.show_explanation:
        st.info(q["explanation"])

        reason_choice_after = st.radio(
            tr("reason_after_prompt"),
            reason_labels,
            index=None,
            key=f"reason_choice_after_{current_idx}",
        )
        if reason_choice_after:
            reason_category_after = reason_label_to_category_loc[reason_choice_after]
            if reason_category_after == "other":
                reason_text_after = st.text_input(
                    tr("please_specify"),
                    key=f"reason_other_after_{current_idx}",
                ).strip()
            else:
                reason_text_after = reason_choice_after

    col1, col2 = st.columns(2)

    if not st.session_state.show_explanation:
        if not reason_choice_before:
            col1.button(tr("explanation_btn"), disabled=True, help=tr("explanation_help"), key=f"explanation_btn_disabled_{current_idx}")
        elif col1.button(tr("explanation_btn"), key=f"explanation_btn_{current_idx}"):
            st.session_state.show_explanation = True
            st.rerun()
    else:
        # Explanation is already shown; keep the button visible but inert.
        col1.button(tr("explanation_btn"), disabled=True, key=f"explanation_btn_shown_{current_idx}")

    is_last = (current_idx + 1 >= len(order))
    next_label = tr("see_results_btn") if is_last else tr("next_btn")
    if st.session_state.show_explanation:
        if not reason_choice_after:
            col2.caption(tr("answer_above_caption"))
        elif col2.button(next_label, key=f"next_btn_{current_idx}"):
            if st.session_state.answer_log:
                st.session_state.answer_log[-1]["reason_category_before"] = reason_category_before
                st.session_state.answer_log[-1]["reason_text_before"] = reason_text_before
                st.session_state.answer_log[-1]["reason_category_after"] = reason_category_after
                st.session_state.answer_log[-1]["reason_text_after"] = reason_text_after
            st.session_state.question_index += 1
            st.session_state.answered = False
            st.session_state.show_explanation = False
            st.rerun()

# Trigger the correct-answer effect right after answering (kept separate from rerun)
if st.session_state.answered and st.session_state.get("show_celebration"):
    cel = STREAK_CELEBRATION if st.session_state.consecutive_correct >= 3 else st.session_state.get("celebration", CELEBRATIONS[0])
    if cel["fx"] == "balloons":
        st.balloons()
    else:
        st.snow()
    st.session_state.show_celebration = False
