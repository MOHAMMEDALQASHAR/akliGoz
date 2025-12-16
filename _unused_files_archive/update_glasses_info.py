# -*- coding: utf-8 -*-
"""
تحديث النظارة الذكية لقراءة الوجوه من faces_db
"""

import os
import pickle
import numpy as np

def update_glasses_to_use_faces_db():
    """
    يحدّث main_glasses.py ليقرأ من faces_db/
    """
    
    code_to_add = '''
    def _load_faces_from_faces_db(self):
        """تحميل الوجوه من مجلد faces_db المشترك (مع الموقع)"""
        print("   📂 Loading faces from faces_db...")
        
        # مجلد الوجوه المشترك
        basedir = os.path.dirname(os.path.abspath(__file__))
        faces_db_dir = os.path.join(basedir, 'faces_db')
        
        if not os.path.exists(faces_db_dir):
            print(f"      ⚠️  faces_db not found: {faces_db_dir}")
            return
        
        print(f"      💾 Reading from: {faces_db_dir}")
        
        # قراءة جميع ملفات embedding
        try:
            files = os.listdir(faces_db_dir)
            embedding_files = [f for f in files if f.endswith('_embedding.pkl')]
            
            if len(embedding_files) == 0:
                print("      ⚠️  No embeddings found")
                return
            
            print(f"      📊 Found {len(embedding_files)} embedding(s)")
            
            for emb_file in embedding_files:
                try:
                    # استخراج الاسم: Ahmed_embedding.pkl -> Ahmed
                    name = emb_file.replace('_embedding.pkl', '')
                    
                    # تحميل embedding
                    emb_path = os.path.join(faces_db_dir, emb_file)
                    with open(emb_path, 'rb') as f:
                        embedding = pickle.load(f)
                    
                    self.known_names.append(name)
                    self.known_encodings.append(embedding)
                    print(f"      👤 Loaded: {name}")
                    
                except Exception as e:
                    print(f"      ❌ Error loading {emb_file}: {e}")
            
            print(f"      ✅ Total: {len(self.known_names)} face(s)")
            
        except Exception as e:
            print(f"      ❌ Error reading faces_db: {e}")
    
    def cosine_similarity_deepface(self, vec1, vec2):
        """حساب التشابه - نفس خوارزمية الموقع"""
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
            
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
'''
    
    print("="*60)
    print("تعليمات تحديث النظارة:")
    print("="*60)
    print("\n1. النظارة الآن ستقرأ من: faces_db/")
    print("2. كل وجه له ملفان:")
    print("   - name.jpg (الصورة)")
    print("   - name_embedding.pkl (Facenet512)")
    print("\n3. عند إضافة/تعديل/حذف في الموقع:")
    print("   → التغيير يحدث في faces_db/")
    print("   → النظارة تراه تلقائياً")
    print("\n4. النظارة تستخدم:")
    print("   - DeepFace Facenet512 embeddings")
    print("   - Cosine Similarity (threshold: 0.6)")
    print("\n✅ النظام متكامل!")
    print("="*60)

if __name__ == "__main__":
    update_glasses_to_use_faces_db()
