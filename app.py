import os, re, json, io, fitz, docx, numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, make_response
from werkzeug.utils import secure_filename
from PIL import Image
from deep_translator import GoogleTranslator
from flask_cors import CORS
import easyocr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indicbart", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indicbart")
reader = easyocr.Reader(['en', 'hi', 'mr'])

app = Flask(__name__)
CORS(app)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
USERS_FILE = "users.json"


# User helpers
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Routes
@app.route('/')
def login_page():
    return render_template("login.html")

@app.route('/register-page')
def register_page():
    return render_template("register.html")

@app.route('/index')
def index_page():
    if 'user_name' not in session:
        flash("Please log in to access the dashboard.")
        return redirect(url_for("login_page"))
    return render_template("index.html", username=session['user_name'])

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route('/register', methods=["POST"])
def register():
    users = load_users()
    data = request.form
    fields = ['first_name', 'last_name', 'mobile', 'email', 'password', 'confirm_password']

    # Check if all fields are filled
    if any(data.get(f, '').strip() == '' for f in fields):
        flash("\u26a0 All fields required")
        return redirect(url_for('register_page'))

    email = data['email'].strip()
    mobile = data['mobile'].strip()

    # Check if email or mobile already exists
    for existing_email, user_data in users.items():
        if email == existing_email:
            flash("⚠ Email already registered.")
            return redirect(url_for('register_page'))
        if mobile == user_data.get('mobile'):
            flash("⚠ Mobile number already registered.")
            return redirect(url_for('register_page'))

    # Save new user if no duplicates
    users[email] = {
        "first_name": data['first_name'],
        "last_name": data['last_name'],
        "mobile": mobile,
        "password": data['password']
    }
    save_users(users)
    flash("✅ Registered! Please log in.")
    return redirect(url_for("login_page"))


@app.route('/check-duplicate', methods=['POST'])
def check_duplicate():
    users = load_users()
    data = request.get_json()
    email = data.get('email', '').strip()
    mobile = data.get('mobile', '').strip()

    for existing_email, user in users.items():
        if email and email == existing_email:
            return jsonify({'status': 'error', 'field': 'email', 'message': 'Email already registered.'})
        if mobile and mobile == user.get('mobile'):
            return jsonify({'status': 'error', 'field': 'mobile', 'message': 'Mobile number already registered.'})
    return jsonify({'status': 'ok'})


@app.route('/login', methods=["POST"])
def login():
    users = load_users()
    inputv = request.form.get("email_or_mobile", "").strip()
    pwd = request.form.get("password", "").strip()
    remember = request.form.get("remember")

    for uemail, u in users.items():
        if inputv in [uemail, u.get("mobile")] and u.get("password") == pwd:
            session['user_name'] = u.get('first_name', 'User')  # ✅ updated here
            resp = make_response(redirect(url_for("index_page")))
            if remember:
                resp.set_cookie('email', inputv, max_age=60*60*24*30)
            flash("Login successful!")
            return resp

    flash("Invalid credentials.")
    return redirect(url_for("login_page"))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login_page"))


# OCR/Translate/Summarize remain same...
@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': '\u26a0 No file provided'}), 400
    file = request.files['file']
    filename = file.filename.lower()
    extracted_text = ""

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            arr = np.array(img)
            result = reader.readtext(arr, detail=0, paragraph=True)
            extracted_text = "\n".join(result).strip()

        elif filename.endswith('.pdf'):
            data_pdf = file.read()
            pdf = fitz.open(stream=data_pdf, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            arr = np.array(img)
            result = reader.readtext(arr, detail=0, paragraph=True)
            extracted_text = "\n".join(result).strip()

        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file.read()))
            extracted_text = "\n".join([p.text for p in doc.paragraphs]).strip()
        else:
            return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        print("Extraction error:", e)
        return jsonify({"error": "Error during extraction"}), 500

    if not extracted_text:
        return jsonify({"error": "No readable text found"}), 400

    session['extracted_text'] = extracted_text
    return jsonify({'text': extracted_text})

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text', '').strip()
    target = data.get('target_language', '').strip()
    if not text or not target:
        return jsonify({"error": "Text or language missing"}), 400

    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        translated_sentences = []
        for sent in sentences:
            translated_sentences.append(GoogleTranslator(source='auto', target=target).translate(sent))
        translated_text = " ".join(translated_sentences).strip()
        session['translated_text'] = translated_text
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": f"Translation error: {e}"}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        input_text = data.get("text", "").strip()

        if not input_text:
            return jsonify({"error": "No text to summarize"}), 400

        # Clean and tokenize
        input_text = re.sub(r'\s+', ' ', input_text)
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids

        # Generate summary
        summary_ids = model.generate(
            input_ids,
            max_length=100,
            min_length=30,
            repetition_penalty=5.0,
            num_beams=15,
            top_p=0.92,
            temperature=1.3,
            do_sample=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": f"Summarization error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
