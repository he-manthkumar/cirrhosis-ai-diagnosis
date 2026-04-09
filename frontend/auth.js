const API_BASE_URL = 'http://127.0.0.1:8000';
const TOKEN_KEY = 'cirrhosis_auth_token';

function setStatus(text, isError = false) {
    const status = document.getElementById('authStatus');
    if (!status) return;
    status.textContent = text;
    status.style.color = isError ? '#b91c1c' : '#0f766e';
}

async function submitAuth(endpoint) {
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;

    if (!username || !password) {
        setStatus('Enter username and password', true);
        return;
    }

    try {
        setStatus('Please wait...');
        const response = await fetch(`${API_BASE_URL}/auth/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || `${endpoint} failed`);
        }

        localStorage.setItem(TOKEN_KEY, payload.access_token);
        setStatus('Success. Redirecting...');
        window.location.href = 'index.html';
    } catch (error) {
        setStatus(error.message, true);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const signupBtn = document.getElementById('signupBtn');
    const loginBtn = document.getElementById('loginBtn');

    if (signupBtn) {
        signupBtn.addEventListener('click', () => submitAuth('signup'));
    }

    if (loginBtn) {
        loginBtn.addEventListener('click', () => submitAuth('login'));
    }
});
