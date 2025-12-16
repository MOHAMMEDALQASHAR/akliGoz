# -*- coding: utf-8 -*-
"""
Enhanced Currency Recognition Module
This file contains the improved identify_currency method with better preprocessing
"""

import cv2
import numpy as np

def preprocess_currency_image(frame):
    """
    Enhanced image preprocessing for better currency recognition
    
    Steps:
    1. Resize to optimal size (320x320)
    2. Denoise for clearer image
    3. Enhance contrast using CLAHE
    """
    # 1. Resize to model's expected size with padding to maintain aspect ratio
    h, w = frame.shape[:2]
    target_size = 320  # Match training size
    
    # Calculate scaling factor
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Enhance image quality
    # Slight denoising
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def identify_currency_enhanced(frame, currency_model, voice, currency_matcher):
    """
    Enhanced currency identification with multiple prediction attempts
    
    Improvements:
    - Preprocesses images for better quality
    - Tries multiple angles (normal, flipped, cropped)
    - Uses ensemble prediction from multiple attempts
    - Higher confidence threshold for more accurate results
    """
    voice.speak(None, use_cache_key="currency_scan")
    
    # --- Option A: Custom Deep Learning Classifier (Best) ---
    if currency_model:
        print("Scanning with Custom Classifier...")
        
        # Preprocess image for better recognition
        processed_frame = preprocess_currency_image(frame)
        
        # Try multiple predictions with slight variations for robustness
        predictions = []
        
        # 1. Original processed image
        result1 = currency_model.predict(processed_frame, verbose=False, imgsz=320)
        if result1 and len(result1) > 0:
            predictions.append(result1[0])
        
        # 2. Horizontally flipped (in case banknote is upside down)
        flipped = cv2.flip(processed_frame, 1)
        result2 = currency_model.predict(flipped, verbose=False, imgsz=320)
        if result2 and len(result2) > 0:
            predictions.append(result2[0])
        
        # 3. Center crop (focus on the center of the note)
        h, w = processed_frame.shape[:2]
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        if h > 2*margin_h and w > 2*margin_w:
            cropped = processed_frame[margin_h:h-margin_h, margin_w:w-margin_w]
            if cropped.size > 0:
                result3 = currency_model.predict(cropped, verbose=False, imgsz=320)
                if result3 and len(result3) > 0:
                    predictions.append(result3[0])
        
        # Find the best prediction across all attempts
        best_conf = 0
        best_cls_name = None
        
        for idx, pred in enumerate(predictions):
            probs = pred.probs
            top1_index = probs.top1
            conf = probs.top1conf.item()
            cls_name = pred.names[top1_index]
            
            print(f"  Attempt {idx+1}: {cls_name} ({conf:.2f})")
            
            if conf > best_conf:
                best_conf = conf
                best_cls_name = cls_name
                
                # Print top 3 for best prediction
                if idx == 0 and hasattr(probs, 'top5'):
                    for i in range(1, min(3, len(probs.top5))):
                        alt_idx = probs.top5[i]
                        print(f"      Alt {i}: {pred.names[alt_idx]} ({probs.data[alt_idx]:.2f})")
        
        # Decision making with improved threshold
        if best_conf > 0.60:  # Increased threshold from 0.50 to 0.60 for more confidence
            msg = f"{best_cls_name} Turk Lirasi"
            print(f"RESULT: {msg} (Confidence: {best_conf:.2f})")
            voice.speak(msg)
            return
        elif best_conf > 0.45:  # Medium confidence - inform user but with caution
            msg = f"Muhtemelen {best_cls_name} Turk Lirasi"  # "Probably X TL"
            print(f"RESULT (Medium Confidence): {msg} ({best_conf:.2f})")
            voice.speak(msg)
            return
    
    # --- Option B: SIFT Feature Matching (Fallback) ---
    print("Fallback to SIFT...")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check Normal
    val = currency_matcher.match(gray)
    
    # Check Mirrored (Flip Horizontal) if no match
    if not val:
       mirrored_gray = cv2.flip(gray, 1)
       val = currency_matcher.match(mirrored_gray)
    
    if val:
         msg = f"{val} Turk Lirasi"
         print(f"RESULT: {msg}")
         voice.speak(msg)
    else:
         voice.speak(None, use_cache_key="no_money")
