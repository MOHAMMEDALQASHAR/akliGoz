# -*- coding: utf-8 -*-
import pygame
import os
import time

print("Testing Audio System...")

try:
    # Initialize pygame mixer
    pygame.mixer.init()
    print("Pygame mixer initialized successfully")
    
    # Test playing a cached audio file
    test_file = "audio_cache/scanning.mp3"
    
    if os.path.exists(test_file):
        print(f"Playing: {test_file}")
        pygame.mixer.music.load(test_file)
        pygame.mixer.music.play()
        
        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        print("Audio played successfully!")
    else:
        print(f"File not found: {test_file}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
