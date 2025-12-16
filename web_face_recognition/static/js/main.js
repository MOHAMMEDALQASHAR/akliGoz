// Main JavaScript for login/register pages
document.addEventListener('DOMContentLoaded', function () {
    // Add smooth animations to form inputs
    const inputs = document.querySelectorAll('input');

    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function () {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
    });

    // Form validation
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const email = this.querySelector('input[type="email"]');
            const password = this.querySelector('input[type="password"]');

            if (email && !isValidEmail(email.value)) {
                e.preventDefault();
                showError(email, 'Lütfen geçerli bir e-posta adresi girin');
                return;
            }

            if (password && password.value.length < 6) {
                e.preventDefault();
                showError(password, 'Şifre en az 6 karakter olmalıdır');
                return;
            }
        });
    });
});

function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function showError(input, message) {
    const formGroup = input.parentElement;

    // Remove existing error
    const existingError = formGroup.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    // Add new error
    const errorDiv = document.createElement('small');
    errorDiv.className = 'error-message';
    errorDiv.style.color = '#ef4444';
    errorDiv.textContent = message;
    formGroup.appendChild(errorDiv);

    // Shake animation
    input.style.animation = 'shake 0.3s';
    setTimeout(() => {
        input.style.animation = '';
    }, 300);
}

// Add shake animation
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
`;
document.head.appendChild(style);
