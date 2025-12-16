import os
import sys

# -----------------------------------------------------------
# HAILO MODEL COMPILER SCRIPT
# Run this ON THE RASPBERRY PI (or Linux PC with Hailo SDK)
# -----------------------------------------------------------

def compile_onnx_to_hef(onnx_path, hef_path):
    print(f"\n🔨 Processing: {onnx_path}...")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: File not found: {onnx_path}")
        return

    try:
        from hailo_sdk_client import ClientRunner
        
        # 1. Initialize Runner for Hailo-8L (RPi 5 AI Kit)
        print("   Initializing Hailo SDK...")
        runner = ClientRunner(hw_arch='hailo8l')
        
        # 2. Parse ONNX
        print("   Parsing ONNX...")
        hn, npz = runner.translate_onnx_model(
            onnx_path, 
            onnx_path.replace('.onnx', ''),
            start_node_names=['images'],
            end_node_names=['output0'], # YOLOv8 standard output
            net_input_shapes={'images': [1, 3, 640, 640]} # YOLO standard size
        )
        
        # 3. Optimize (Quantization) - Uses default calibration for simplicity
        # Note: Real calibration needs images, this assumes Model Zoo style
        print("   Optimizing (Quantization)...")
        runner.optimize_full_precision() 
        
        # 4. Compile
        print("   Compiling to HEF...")
        hef = runner.compile()
        
        # 5. Save
        with open(hef_path, 'wb') as f:
            f.write(hef)
        print(f"✅ Success! Saved to: {hef_path}")
        
    except ImportError:
        print("❌ CRITICAL ERROR: 'hailo_sdk_client' not installed.")
        print("   Please install the Hailo Dataflow Compiler (DFC) on this machine.")
    except Exception as e:
        print(f"❌ Error during compilation: {e}")

if __name__ == "__main__":
    print("🚀 Hailo Model Compiler Script")
    print("------------------------------")
    
    # 1. Compile YOLO
    compile_onnx_to_hef("yolov8n.onnx", "yolov8n.hef")
    
    # 2. Compile Currency (if exists)
    if os.path.exists("currency_model.onnx"):
        compile_onnx_to_hef("currency_model.onnx", "currency_model.hef")
    
    # 3. Compile Color (if exists)
    if os.path.exists("color_model.onnx"):
        compile_onnx_to_hef("color_model.onnx", "color_model.hef")
    
    print("\n------------------------------")
    print("Done. If successful, use the .hef files with main_glasses_hailo.py")
