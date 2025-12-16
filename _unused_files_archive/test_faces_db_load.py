# Test: Load faces from faces_db
import os
import pickle

basedir = os.path.dirname(os.path.abspath(__file__))
faces_db_dir = os.path.join(basedir, 'faces_db')

print("="*60)
print("Testing faces_db loading")
print("="*60)
print(f"Directory: {faces_db_dir}")
print(f"Exists: {os.path.exists(faces_db_dir)}")

if os.path.exists(faces_db_dir):
    files = os.listdir(faces_db_dir)
    print(f"\nFiles found: {len(files)}")
    
    for f in files:
        print(f"  - {f}")
        if f.endswith('_embedding.pkl'):
            name = f.replace('_embedding.pkl', '')
            print(f"    Name: {name}")
            
            emb_path = os.path.join(faces_db_dir, f)
            try:
                with open(emb_path, 'rb') as file:
                    emb = pickle.load(file)
                print(f"    Embedding size: {len(emb)}")
                print(f"    Type: {type(emb)}")
            except Exception as e:
                print(f"    Error: {e}")

print("\n" + "="*60)
