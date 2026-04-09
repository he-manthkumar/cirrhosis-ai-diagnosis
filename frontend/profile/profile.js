const API_BASE_URL = 'http://127.0.0.1:8000';
const TOKEN_KEY = 'cirrhosis_auth_token';

const profileUsername = document.getElementById('profileUsername');
const searchHistoryBtn = document.getElementById('searchHistoryBtn');
const searchNameInput = document.getElementById('search_name');
const historyResults = document.getElementById('historyResults');
const logoutBtn = document.getElementById('logoutBtn');

function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

function getAuthHeaders() {
    const token = getToken();
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    return headers;
}

function ensureAuthenticated() {
    if (!getToken()) {
        window.location.href = '../login.html';
    }
}

function parseJwtPayload(token) {
    try {
        const base64Url = token.split('.')[1];
        if (!base64Url) return null;
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const json = decodeURIComponent(
            atob(base64)
                .split('')
                .map((c) => `%${(`00${c.charCodeAt(0).toString(16)}`).slice(-2)}`)
                .join('')
        );
        return JSON.parse(json);
    } catch (error) {
        return null;
    }
}

function getStatusLabel(status) {
    const labels = {
        C: 'Stable - Low Mortality Risk',
        CL: 'Critical - High Mortality Risk',
        D: 'Death Imminent Without Intervention'
    };
    return labels[status] || status;
}

async function fetchHistory() {
    const name = searchNameInput.value.trim();
    if (!name) {
        alert('Please enter a name to search');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/predict/history/${encodeURIComponent(name)}`, {
            headers: getAuthHeaders()
        });

        if (response.status === 404) {
            historyResults.style.display = 'block';
            historyResults.innerHTML = '<p class="no-data">No records found for this patient.</p>';
            return;
        }

        if (!response.ok) {
            throw new Error('Failed to fetch history');
        }

        const records = await response.json();

        historyResults.style.display = 'block';
        historyResults.innerHTML = records.map((record) => `
            <div class="history-item">
                <strong>Date:</strong> ${new Date(record.created_at + 'Z').toLocaleString()}<br>
                <strong>Prediction:</strong> ${getStatusLabel(record.prediction)} (${(record.confidence * 100).toFixed(1)}% confidence)<br>
                <strong>Clinical Notes:</strong> ${record.narrative ? record.narrative.substring(0, 150) + '...' : 'N/A'}
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer; color: #0b6a67;">View Full Input Data</summary>
                    <pre>${JSON.stringify(record.clinical_data, null, 2)}</pre>
                </details>
            </div>
        `).join('');
    } catch (error) {
        alert(`Error fetching history: ${error.message}`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    ensureAuthenticated();

    const token = getToken();
    const payload = token ? parseJwtPayload(token) : null;
    if (profileUsername) {
        profileUsername.textContent = payload?.sub || 'Unknown user';
    }

    if (searchHistoryBtn) {
        searchHistoryBtn.addEventListener('click', fetchHistory);
    }

    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            localStorage.removeItem(TOKEN_KEY);
            window.location.href = '../login.html';
        });
    }
});
