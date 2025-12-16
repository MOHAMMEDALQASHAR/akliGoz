# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import json
import pickle
from deepface import DeepFace  # Facenet512 للتعرف على الوجوه

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# --- CONFIGURATION (ABSOLUTE PATHS) ---
# Get the directory where app.py is located: .../web_face_recognition
current_dir = os.path.abspath(os.path.dirname(__file__))

# 1. Database in web_face_recognition/instance/face_recognition.db (Standard Flask)
instance_path = os.path.join(current_dir, 'instance')
os.makedirs(instance_path, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///' + os.path.join(instance_path, 'face_recognition.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 2. Faces DB - مجلد مشترك مع النظارة الذكية
basedir = os.path.dirname(current_dir)
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'faces_db')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


db = SQLAlchemy(app)
oauth = OAuth(app)

# Google OAuth Configuration
google = oauth.register(
    name='google',
    client_id='YOUR_GOOGLE_CLIENT_ID',
    client_secret='YOUR_GOOGLE_CLIENT_SECRET',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200))
    name = db.Column(db.String(100))
    google_id = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    encoding = db.Column(db.LargeBinary, nullable=False)  # Store as binary
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# Helper Functions
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def extract_face_features_deepface(image):
    """استخراج ميزات الوجه باستخدام DeepFace Facenet512 (512-dimensional embedding)"""
    try:
        # تحويل الصورة إلى تنسيق مؤقت
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, image)
        
        # استخراج embedding باستخدام Facenet512
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet512',
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # حذف الملف المؤقت
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if embedding_objs and len(embedding_objs) > 0:
            embedding = np.array(embedding_objs[0]['embedding'])
            print(f"Extracted Facenet512 embedding: shape={embedding.shape}")
            return embedding
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting features: {e}")
        if os.path.exists("temp_face.jpg"):
            os.remove("temp_face.jpg")
        return None

def cosine_similarity(vec1, vec2):
    """حساب التشابه الجيبي بين vectorين (Cosine Similarity)"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    try:
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        print(f"Error in cosine_similarity: {e}")
        return 0.0

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.password and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Hatalı giriş bilgileri')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Bu e-posta adresi zaten kayıtlı')
        
        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session['user_name'] = new_user.name
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if user_info:
            user = User.query.filter_by(google_id=user_info['sub']).first()
            
            if not user:
                user = User(
                    email=user_info['email'],
                    name=user_info.get('name', user_info['email']),
                    google_id=user_info['sub']
                )
                db.session.add(user)
                db.session.commit()
            
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Google login error: {e}")
    
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    faces = Face.query.filter_by(user_id=user_id).all()
    return render_template('dashboard.html', faces=faces, user_name=session.get('user_name'))

@app.route('/add_face', methods=['POST'])
@login_required
def add_face():
    try:
        print("DEBUG: add_face called")
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            print("DEBUG: Missing name or image data")
            return jsonify({'success': False, 'error': 'Ad ve resim gerekli'})
        
        print(f"DEBUG: Processing image for {name}")
        
        # Decode base64 image
        try:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"DEBUG: Image decode error: {e}")
            return jsonify({'success': False, 'error': 'Resim işlenemedi'})
            
        if img is None:
             print("DEBUG: img is None after decode")
             return jsonify({'success': False, 'error': 'Resim boş'})

        # Extract face features using DeepFace Facenet512
        print("DEBUG: Extracting Facenet512 embedding...")
        features = extract_face_features_deepface(img)
        if features is None:
            print("DEBUG: No face detected in image")
            return jsonify({'success': False, 'error': 'Resimde yüz bulunamadı - Lütfen kameraya daha yakın durun veya ışığı kontrol edin'})
        
        print("DEBUG: Face detected. Saving...")
        
        # Save image
        user_id = session['user_id']
        # استخدام اسم الشخص كاسم للملف
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        filename = f"{safe_name}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"DEBUG: Saving file to {filepath}")
        cv2.imwrite(filepath, img)
        
        # حفظ embedding في ملف منفصل للنظارة
        embedding_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe_name}_embedding.pkl")
        with open(embedding_file, 'wb') as f:
            pickle.dump(features, f)
        
        # Save to database
        encoding_bytes = pickle.dumps(features)
        new_face = Face(
            user_id=user_id,
            name=name,
            image_path=filepath,
            encoding=encoding_bytes
        )
        db.session.add(new_face)
        db.session.commit()
        print("DEBUG: Database commit successful")
        
        # Prepare response data
        print(f"DEBUG: Saved {name} to faces_db!")
        created_at = new_face.created_at.strftime('%Y-%m-%d %H:%M')
        
        return jsonify({
            'success': True, 
            'message': f'{name} başarıyla kaydedildi - النظارة تستطيع رؤيته الآن!',
            'face': {
                'id': new_face.id,
                'name': new_face.name,
                'image_url': url_for('view_face', filename=filename),
                'created_at': created_at
            }
        })
    
    except Exception as e:
        print(f"DEBUG: Critical Error in add_face: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/recognize_face', methods=['POST'])
@login_required
def recognize_face():
    try:
        image_data = request.form.get('image')
        user_id = session['user_id']
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Resim gerekli'})
        
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract face
        face_roi = extract_face_features(img)
        if face_roi is None:
            return jsonify({'success': True, 'name': None, 'message': 'Yüz bulunamadı'})
        
        # Get all faces for this user
        saved_faces = Face.query.filter_by(user_id=user_id).all()
        
        if not saved_faces:
            return jsonify({'success': True, 'name': None, 'message': 'Kayıtlı yüz yok'})
        
        # Create LBPH recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Prepare training data
        faces_list = []
        labels_list = []
        names_dict = {}
        
        for idx, face in enumerate(saved_faces):
            if face.encoding:
                try:
                    face_img = pickle.loads(face.encoding)
                    # Ensure it's 2D image
                    if len(face_img.shape) == 1:
                        face_img = face_img.reshape(200, 200)
                    faces_list.append(face_img)
                    labels_list.append(idx)
                    names_dict[idx] = face.name
                except:
                    pass
        
        if len(faces_list) == 0:
            return jsonify({'success': True, 'name': None, 'message': 'Kayıtlı yüz yok'})
        
        # Train recognizer
        recognizer.train(faces_list, np.array(labels_list))
        
        # Predict
        label, confidence = recognizer.predict(face_roi)
        
        # Lower confidence = better match (distance metric)
        # Threshold: if confidence < 50, it's a good match
        if confidence < 70:
            name = names_dict.get(label, 'Bilinmeyen')
            return jsonify({
                'success': True,
                'name': name,
                'message': f'Merhaba {name}!',
                'confidence': float(confidence)
            })
        else:
            return jsonify({'success': True, 'name': None, 'message': 'Yüz tanınmadı'})
        
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/view_face/<filename>')
@login_required
def view_face(filename):
    """عرض صورة من مجلد faces_db المشترك"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving image: {e}")
        return str(e), 500

@app.route('/delete_face/<int:face_id>', methods=['POST'])
@login_required
def delete_face(face_id):
    try:
        user_id = session['user_id']
        face = Face.query.filter_by(id=face_id, user_id=user_id).first()
        
        if not face:
            return jsonify({'success': False, 'error': 'Yüz bulunamadı'})
        
        # Delete image file from faces_db
        if face.image_path and os.path.exists(face.image_path):
            try:
                os.remove(face.image_path)
                print(f"Deleted image: {face.image_path}")
            except Exception as e:
                print(f"Error deleting image: {e}")
        
        # Delete embedding file (name_embedding.pkl)
        if face.image_path:
            # Extract filename without extension
            base_name = os.path.splitext(os.path.basename(face.image_path))[0]
            embedding_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_embedding.pkl")
            if os.path.exists(embedding_path):
                try:
                    os.remove(embedding_path)
                    print(f"Deleted embedding: {embedding_path}")
                except Exception as e:
                    print(f"Error deleting embedding: {e}")
        
        # Delete from database
        db.session.delete(face)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Başarıyla silindi'})
    
    except Exception as e:
        print(f"Delete error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/edit_face/<int:face_id>', methods=['POST'])
@login_required
def edit_face(face_id):
    try:
        user_id = session['user_id']
        face = Face.query.filter_by(id=face_id, user_id=user_id).first()
        
        if not face:
            return jsonify({'success': False, 'error': 'Yüz bulunamadı'})
            
        new_name = request.form.get('name')
        if not new_name:
            return jsonify({'success': False, 'error': 'Yeni isim gerekli'})
        
        old_name = face.name
        
        # Rename files in faces_db if name changed
        if face.image_path and os.path.exists(face.image_path):
            try:
                # Get old file paths
                old_image_path = face.image_path
                old_base_name = os.path.splitext(os.path.basename(old_image_path))[0]
                old_embedding_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{old_base_name}_embedding.pkl")
                
                # Create new file names
                safe_new_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe_new_name}.jpg")
                new_embedding_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe_new_name}_embedding.pkl")
                
                # Rename image file
                if os.path.exists(old_image_path):
                    os.rename(old_image_path, new_image_path)
                    face.image_path = new_image_path
                    print(f"Renamed image: {old_image_path} -> {new_image_path}")
                
                # Rename embedding file
                if os.path.exists(old_embedding_path):
                    os.rename(old_embedding_path, new_embedding_path)
                    print(f"Renamed embedding: {old_embedding_path} -> {new_embedding_path}")
                    
            except Exception as e:
                print(f"Error renaming files: {e}")
                return jsonify({'success': False, 'error': f'Dosya yeniden adlandırılamadı: {str(e)}'})
        
        # Update name in database
        face.name = new_name
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'İsim güncellendi'})
    
    except Exception as e:
        print(f"Edit error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("="*60)
    print("Yuz Tanima Sistemi - Face Recognition System")
    print("="*60)
    print(f"Web sitesi mevcut: http://localhost:5000")
    print(f"Website available at: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
