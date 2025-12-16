# تشغيل نظام التعرف على الوجوه
# Run Face Recognition System

Write-Host "================================" -ForegroundColor Cyan
Write-Host "نظام التعرف على الوجوه" -ForegroundColor Green
Write-Host "Face Recognition System" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-Not (Test-Path "app.py")) {
    Write-Host "خطأ: الرجاء تشغيل هذا السكريبت من مجلد web_face_recognition" -ForegroundColor Red
    Write-Host "Error: Please run this script from web_face_recognition folder" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host "إنشاء بيئة افتراضية..." -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "تفعيل البيئة الافتراضية..." -ForegroundColor Yellow
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install requirements
Write-Host ""
Write-Host "تثبيت المكتبات المطلوبة..." -ForegroundColor Yellow
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create directories
Write-Host ""
Write-Host "إنشاء المجلدات الضرورية..." -ForegroundColor Yellow
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "uploads\faces" | Out-Null
New-Item -ItemType Directory -Force -Path "instance" | Out-Null

# Run the app
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "تشغيل التطبيق..." -ForegroundColor Green
Write-Host "Starting application..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "الموقع متاح على: http://localhost:5000" -ForegroundColor Green
Write-Host "Website available at: http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "اضغط Ctrl+C لإيقاف التطبيق" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

python app.py
