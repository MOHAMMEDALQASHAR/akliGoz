// Dashboard JavaScript
let video = null;
let canvas = null;
let stream = null;
let capturedImageData = null;
let recognitionInterval = null;

// ===== GLOBAL HANDLERS - Must be defined OUTSIDE DOMContentLoaded =====
// These functions are called directly from HTML onclick attributes

// Variables to store current operation context
let currentEditButton = null;
let currentDeleteButton = null;

window.handleEditClick = function (btn) {
    console.log('EDIT CLICKED via inline handler');
    currentEditButton = btn;
    const currentName = btn.dataset.faceName;

    // Show modal
    const modal = document.getElementById('editModal');
    const input = document.getElementById('editNameInput');
    input.value = currentName;
    modal.classList.add('show');

    // Focus input
    setTimeout(() => input.focus(), 100);
};

window.confirmEdit = function () {
    if (!currentEditButton) return;

    const input = document.getElementById('editNameInput');
    const newName = input.value.trim();
    const currentName = currentEditButton.dataset.faceName;
    const faceId = currentEditButton.dataset.faceId;

    if (!newName) {
        alert('❌ İsim boş olamaz!');
        return;
    }

    if (newName === currentName) {
        cancelEdit();
        return;
    }

    const formData = new FormData();
    formData.append('name', newName);

    fetch(`/edit_face/${faceId}`, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                alert('✅ İsim başarıyla güncellendi: ' + newName);
                const card = currentEditButton.closest('.face-card');
                card.querySelector('h4').textContent = newName;
                currentEditButton.dataset.faceName = newName;
                cancelEdit();
            } else {
                alert('❌ Hata: ' + (result.error || 'Güncelleme başarısız'));
            }
        })
        .catch(error => {
            console.error('Edit error:', error);
            alert('❌ Güncelleme sırasında hata oluştu');
        });
};

window.cancelEdit = function () {
    const modal = document.getElementById('editModal');
    modal.classList.remove('show');
    currentEditButton = null;
};

window.handleDeleteClick = function (btn) {
    console.log('DELETE CLICKED via inline handler');
    currentDeleteButton = btn;

    // Show modal
    const modal = document.getElementById('deleteModal');
    modal.classList.add('show');
};

window.confirmDelete = function () {
    if (!currentDeleteButton) return;

    const faceId = currentDeleteButton.dataset.faceId;

    fetch(`/delete_face/${faceId}`, {
        method: 'POST'
    })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                const card = currentDeleteButton.closest('.face-card');
                card.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => {
                    card.remove();
                    const grid = document.getElementById('facesGrid');
                    if (!grid || grid.children.length === 0) {
                        if (grid) {
                            grid.innerHTML = `
                            <div class="empty-state">
                                <p>Henüz kaydedilmiş yüz yok</p>
                                <p>Yeni yüz eklemek için kamerayı kullanın</p>
                            </div>
                        `;
                        }
                    }
                }, 300);
                cancelDelete();
                showToast('Başarıyla silindi', 'success');
            } else {
                alert('Hata: ' + result.error);
            }
        })
        .catch(error => {
            console.error('Delete error:', error);
            alert('Silerken bir hata oluştu');
        });
};

window.cancelDelete = function () {
    const modal = document.getElementById('deleteModal');
    modal.classList.remove('show');
    currentDeleteButton = null;
};
// ===== END GLOBAL HANDLERS =====

document.addEventListener('DOMContentLoaded', function () {
    console.log('Dashboard.js loaded! Version: 20251213-v4');

    video = document.getElementById('video');
    canvas = document.getElementById('canvas');

    const startCameraBtn = document.getElementById('startCamera');
    const stopCameraBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('captureBtn');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const addFaceForm = document.getElementById('addFaceForm');

    // Start Camera
    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', startCamera);
    }

    // Stop Camera
    if (stopCameraBtn) {
        stopCameraBtn.addEventListener('click', stopCamera);
    }



    // Capture Photo
    if (captureBtn) {
        captureBtn.addEventListener('click', capturePhoto);
    }

    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', toggleRecognition);
    }

    if (addFaceForm) {
        addFaceForm.addEventListener('submit', saveFace);
    }

    // Initialize Delete and Edit buttons for existing items
    document.querySelectorAll('.delete-face').forEach(btn => {
        btn.addEventListener('click', deleteFace);
    });

    document.querySelectorAll('.edit-face').forEach(btn => {
        btn.addEventListener('click', editFace);
    });
});



async function startCamera() {
    try {
        // Enumerate devices to find the integrated camera and BLOCK iVCam
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        // Strategy: 
        // 1. FILTER OUT any device with "ivcam" or "e2eoft" in the name
        // 2. Look for "integrated", "built-in", "internal"

        const bannedKeywords = ['ivcam', 'e2eoft', 'virtual'];
        const validDevices = videoDevices.filter(device => {
            const label = (device.label || '').toLowerCase();
            return !bannedKeywords.some(bad => label.includes(bad));
        });

        let targetDeviceId = undefined;

        if (validDevices.length > 0) {
            // Look for preferred keywords in VALID devices only
            const preferredKeywords = ['integrated', 'built-in', 'internal', 'webcam', 'front', 'easycamera'];

            for (const device of validDevices) {
                const label = (device.label || '').toLowerCase();
                if (preferredKeywords.some(keyword => label.includes(keyword))) {
                    targetDeviceId = device.deviceId;
                    console.log(`Found PREFERRED camera: ${device.label}`);
                    break;
                }
            }

            // If no preferred keyword found, use the first VALID device
            if (!targetDeviceId) {
                targetDeviceId = validDevices[0].deviceId;
                console.log(`Using fallback VALID camera: ${validDevices[0].label}`);
            }
        } else {
            console.warn("No valid cameras found after filtering!");
        }

        // If we found a specific target, use it. Otherwise, let browser decide (but we tried our best).
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                deviceId: targetDeviceId ? { exact: targetDeviceId } : undefined
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);

        video.srcObject = stream;
        video.play();

        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-block';
        document.getElementById('captureBtn').disabled = false;
        document.getElementById('recognizeBtn').style.display = 'inline-block';

        showToast('Kamera başarıyla başlatıldı', 'success');
    } catch (error) {
        console.error('Camera error:', error);
        showToast('Kamera başlatılamadı: ' + error.message, 'error');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null; // Ensure stream is cleared

        if (recognitionInterval) {
            clearInterval(recognitionInterval);
            recognitionInterval = null;
        }

        document.getElementById('startCamera').style.display = 'inline-block';
        document.getElementById('stopCamera').style.display = 'none';
        document.getElementById('captureBtn').disabled = true;
        document.getElementById('recognizeBtn').style.display = 'none';
        document.getElementById('recognizeBtn').textContent = '🔍 Yüzü Tanı';
        document.getElementById('recognitionResult').classList.remove('show');

        showToast('Kamera durduruldu', 'success');
    }
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    capturedImageData = canvas.toDataURL('image/jpeg');

    const preview = document.getElementById('capturedImagePreview');
    const previewImg = document.getElementById('previewImg');

    previewImg.src = capturedImageData;
    preview.style.display = 'block';

    showToast('Fotoğraf başarıyla çekildi', 'success');
}

async function saveFace(e) {
    e.preventDefault();

    const name = document.getElementById('faceName').value.trim();

    if (!name) {
        showToast('Lütfen kişi adını girin', 'error');
        return;
    }

    if (!capturedImageData) {
        showToast('Lütfen önce fotoğraf çekin', 'error');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('image', capturedImageData);

        const response = await fetch('/add_face', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');

            // Reset form
            document.getElementById('faceName').value = '';
            document.getElementById('capturedImagePreview').style.display = 'none';
            capturedImageData = null;

            // Dynamically add new card
            if (result.face) {
                // Show specific success message
                showToast(`✅ ${result.face.name} başarıyla veritabanına kaydedildi!`, 'success');

                const grid = document.getElementById('facesGrid');

                // Remove empty state if present
                const emptyState = grid.querySelector('.empty-state');
                if (emptyState) {
                    emptyState.remove();
                }

                const card = document.createElement('div');
                card.className = 'face-card';
                card.dataset.faceId = result.face.id;

                card.innerHTML = `
                    <div class="face-image">
                        <img src="${result.face.image_url}" alt="${result.face.name}">
                    </div>
                    <div class="face-info">
                        <h4>${result.face.name}</h4>
                        <small>${result.face.created_at}</small>
                    </div>
                    <div class="face-actions" style="display: flex; gap: 5px; justify-content: center; margin-top: 10px;">
                        <button class="btn btn-primary btn-sm edit-face" data-face-id="${result.face.id}" data-face-name="${result.face.name}">
                            ✏️ Düzenle
                        </button>
                        <button class="btn btn-danger btn-sm delete-face" data-face-id="${result.face.id}">
                            🗑️ Sil
                        </button>
                    </div>
                `;

                // Add animations
                card.style.animation = 'fadeIn 0.5s ease';

                // Append as first child if you want it at top, or last
                // Usually newest first is better for "immediate feedback"
                grid.insertBefore(card, grid.firstChild);

                // Attach listeners
                card.querySelector('.delete-face').addEventListener('click', deleteFace);
                card.querySelector('.edit-face').addEventListener('click', editFace);
            } else {
                // Fallback if no face data returned
                setTimeout(() => window.location.reload(), 500);
            }
        } else {
            showToast(result.error, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Kaydederken bir hata oluştu', 'error');
    }
}

function toggleRecognition() {
    const btn = document.getElementById('recognizeBtn');

    if (recognitionInterval) {
        // Stop recognition
        clearInterval(recognitionInterval);
        recognitionInterval = null;
        btn.textContent = '🔍 Yüzü Tanı';
        document.getElementById('recognitionResult').classList.remove('show');
    } else {
        // Start recognition
        btn.textContent = '⏹️ Tanımayı Durdur';
        recognizeFace(); // Call immediately
        recognitionInterval = setInterval(recognizeFace, 2000); // Every 2 seconds
    }
}

async function recognizeFace() {
    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg');

        const formData = new FormData();
        formData.append('image', imageData);

        const response = await fetch('/recognize_face', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        const resultDiv = document.getElementById('recognitionResult');

        if (result.success && result.name) {
            resultDiv.textContent = result.message;
            resultDiv.style.background = 'rgba(16, 185, 129, 0.9)';
            resultDiv.classList.add('show');

            // Speak the name
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(result.message);
                utterance.lang = 'tr-TR';
                speechSynthesis.speak(utterance);
            }
        } else if (result.success) {
            resultDiv.textContent = result.message;
            resultDiv.style.background = 'rgba(0, 0, 0, 0.8)';
            resultDiv.classList.add('show');

            // Hide after 2 seconds if no face
            setTimeout(() => {
                resultDiv.classList.remove('show');
            }, 2000);
        }
    } catch (error) {
        console.error('Recognition error:', error);
    }
}

async function deleteFace(e) {
    e.preventDefault();
    e.stopPropagation();

    // Use currentTarget to get the button element, not the emoji/text inside
    const btn = e.currentTarget;
    const faceId = btn.dataset.faceId;

    console.log('Delete button clicked, Face ID:', faceId);

    if (!confirm('Bu yüzü silmek istediğinize emin misiniz?')) {
        return;
    }

    try {
        const response = await fetch(`/delete_face/${faceId}`, {
            method: 'POST'
        });

        const result = await response.json();
        console.log('Delete result:', result);

        if (result.success) {
            showToast(result.message, 'success');

            // Remove card from UI
            const card = btn.closest('.face-card');
            card.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => {
                card.remove();

                // Check if grid is empty
                const grid = document.getElementById('facesGrid');
                if (grid.children.length === 0) {
                    grid.innerHTML = `
                        <div class="empty-state">
                            <p>Henüz kaydedilmiş yüz yok</p>
                            <p>Yeni yüz eklemek için kamerayı kullanın</p>
                        </div>
                    `;
                }
            }, 300);
        } else {
            showToast(result.error, 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Silerken bir hata oluştu', 'error');
    }
}

async function editFace(e) {
    e.preventDefault();
    e.stopPropagation();

    // Use currentTarget to get the button element
    const btn = e.currentTarget;
    const faceId = btn.dataset.faceId;
    const currentName = btn.dataset.faceName;

    console.log('Edit button clicked, Face ID:', faceId, 'Current name:', currentName);

    const newName = prompt('Yeni ismi girin:', currentName);

    if (newName && newName.trim() !== '' && newName !== currentName) {
        try {
            const formData = new FormData();
            formData.append('name', newName.trim());

            const response = await fetch(`/edit_face/${faceId}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            console.log('Edit result:', result);

            if (result.success) {
                showToast('İsim başarıyla güncellendi', 'success');

                // Update UI directly
                const card = btn.closest('.face-card');
                card.querySelector('h4').textContent = newName.trim();
                btn.dataset.faceName = newName.trim();

            } else {
                showToast(result.error || 'Güncelleme başarısız', 'error');
            }
        } catch (error) {
            console.error('Edit error:', error);
            showToast('Güncelleme sırasında hata oluştu', 'error');
        }
    }
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Add fadeOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.8); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
