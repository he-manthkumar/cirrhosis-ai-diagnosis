// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// DOM Elements
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImageBtn = document.getElementById('removeImage');

// State
let selectedImageBase64 = null;
let selectedImageMimeType = 'image/jpeg';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupImageUpload();
    setupFormSubmission();
});

// Image Upload Handling
function setupImageUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // File selection
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processImage(file);
        }
    });
    
    // Remove image
    removeImageBtn.addEventListener('click', () => {
        selectedImageBase64 = null;
        selectedImageMimeType = 'image/jpeg';
        imagePreview.style.display = 'none';
        uploadArea.style.display = 'block';
        imageInput.value = '';
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processImage(file);
    }
}

function processImage(file) {
    selectedImageMimeType = file.type;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        // Extract base64 from data URL
        selectedImageBase64 = dataUrl.split(',')[1];
        
        // Show preview
        previewImg.src = dataUrl;
        imagePreview.style.display = 'inline-block';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Form Submission
function setupFormSubmission() {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitPrediction();
    });
}

async function submitPrediction() {
    // Show loading
    loadingOverlay.classList.add('active');
    submitBtn.disabled = true;
    
    try {
        // Gather form data
        const patientData = {
            age: parseInt(document.getElementById('age').value),
            sex: document.getElementById('sex').value,
            drug: document.getElementById('drug').value,
            stage: parseInt(document.getElementById('stage').value),
            bilirubin: parseFloat(document.getElementById('bilirubin').value),
            albumin: parseFloat(document.getElementById('albumin').value),
            alk_phos: parseFloat(document.getElementById('alk_phos').value),
            sgot: parseFloat(document.getElementById('sgot').value),
            cholesterol: parseFloat(document.getElementById('cholesterol').value) || null,
            tryglicerides: parseFloat(document.getElementById('tryglicerides').value) || null,
            copper: parseFloat(document.getElementById('copper').value) || null,
            platelets: parseFloat(document.getElementById('platelets').value) || null,
            prothrombin: parseFloat(document.getElementById('prothrombin').value),
            ascites: document.getElementById('ascites').checked ? 'Y' : 'N',
            hepatomegaly: document.getElementById('hepatomegaly').checked ? 'Y' : 'N',
            spiders: document.getElementById('spiders').checked ? 'Y' : 'N',
            edema: document.getElementById('edema').checked ? 'Y' : 'N'
        };
        
        // Build request
        const requestBody = {
            patient: patientData
        };
        
        // Add image if selected
        if (selectedImageBase64) {
            requestBody.image = {
                image_base64: selectedImageBase64,
                image_mime_type: selectedImageMimeType
            };
        }
        
        // Make API call
        const response = await fetch(`${API_BASE_URL}/predict/full`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        loadingOverlay.classList.remove('active');
        submitBtn.disabled = false;
    }
}

// Display Results
function displayResults(result) {
    resultsSection.style.display = 'flex';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Display prediction
    displayPrediction(result.prediction);
    
    // Display explanation
    displayExplanation(result.explanation);
    
    // Display image analysis if available
    if (result.image_analysis) {
        displayImageAnalysis(result.image_analysis);
    } else {
        document.getElementById('imageAnalysisCard').style.display = 'none';
    }
}

function displayPrediction(prediction) {
    const statusValue = document.getElementById('statusValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const riskBadge = document.getElementById('riskBadge');
    
    // Status
    const status = prediction.final_prediction;
    statusValue.textContent = getStatusLabel(status);
    statusValue.className = `status-value status-${status}`;
    
    // Confidence
    const confidence = (prediction.confidence * 100).toFixed(1);
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;
    
    // Risk Level
    const riskLevel = prediction.risk_level || 'Unknown';
    riskBadge.textContent = riskLevel;
    riskBadge.className = `risk-badge risk-${riskLevel.toLowerCase()}`;
    
    // Probabilities
    displayProbabilities(prediction.probabilities);
    
    // Base Models
    displayBaseModels(prediction.base_model_predictions);
}

function getStatusLabel(status) {
    const labels = {
        'C': 'Stable – Low Mortality Risk',
        'CL': 'Critical – High Mortality Risk',
        'D': 'Death Imminent Without Intervention'
    };
    return labels[status] || status;
}

function displayProbabilities(probabilities) {
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';

    const labels = {
        'C': 'Stable – Low Risk',
        'CL': 'Critical – High Risk',
        'D': 'Death Imminent Without Intervention'
    };

    for (const [key, value] of Object.entries(probabilities)) {
        const percentage = (value * 100).toFixed(1);

        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <span class="prob-label">${labels[key]}</span>
            <div class="prob-bar-container">
                <div class="prob-bar prob-${key}" style="width: ${percentage}%">
                    ${percentage}%
                </div>
            </div>
        `;
        probBars.appendChild(probItem);
    }
}


function displayBaseModels(models) {
    const modelsGrid = document.getElementById('modelsGrid');
    modelsGrid.innerHTML = '';
    
    models.forEach(model => {
        const confidence = (model.confidence * 100).toFixed(1);
        
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.innerHTML = `
            <div class="model-name">${formatModelName(model.model_name)}</div>
            <div class="model-prediction">
                <span>Prediction:</span>
                <strong>${model.prediction}</strong>
            </div>
            <div class="model-prediction">
                <span>Confidence:</span>
                <strong>${confidence}%</strong>
            </div>
        `;
        modelsGrid.appendChild(modelCard);
    });
}

function formatModelName(name) {
    const names = {
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'decision_tree': 'Decision Tree'
    };
    return names[name] || name;
}

function displayExplanation(explanation) {
    // Narrative
    const narrativeText = document.getElementById('narrativeText');
    if (explanation.narrative) {
        narrativeText.textContent = explanation.narrative;
        narrativeText.classList.remove('loading');
    } else {
        narrativeText.innerHTML = '<em>Narrative generation unavailable. Please check API configuration.</em>';
        narrativeText.classList.remove('loading');
    }
    
    // Decision Rules
    displayRules(explanation.rule_path);
    
    // Key Features
    displayFeatures(explanation.key_features);
}

function displayRules(rules) {
    const rulesList = document.getElementById('rulesList');
    rulesList.innerHTML = '';
    
    if (!rules || rules.length === 0) {
        rulesList.innerHTML = '<p class="no-data">No decision rules available</p>';
        return;
    }
    
    rules.forEach((rule, index) => {
        const ruleItem = document.createElement('div');
        ruleItem.className = 'rule-item';
        ruleItem.innerHTML = `
            <span class="rule-number">${index + 1}</span>
            <span class="rule-text">${rule}</span>
        `;
        rulesList.appendChild(ruleItem);
    });
}

function displayFeatures(features) {
    const featuresGrid = document.getElementById('featuresGrid');
    featuresGrid.innerHTML = '';
    
    if (!features || Object.keys(features).length === 0) {
        featuresGrid.innerHTML = '<p class="no-data">No feature information available</p>';
        return;
    }
    
    for (const [name, data] of Object.entries(features)) {
        const featureCard = document.createElement('div');
        featureCard.className = 'feature-card';
        
        const statusClass = data.status.replace(' ', '-').toLowerCase();
        
        featureCard.innerHTML = `
            <div class="feature-name">${name}</div>
            <div class="feature-value">${data.formatted}</div>
            <span class="feature-status ${statusClass}">${data.status}</span>
            <div class="feature-description">${data.description}</div>
        `;
        featuresGrid.appendChild(featureCard);
    }
}

function displayImageAnalysis(imageAnalysis) {
    const imageAnalysisCard = document.getElementById('imageAnalysisCard');
    const imageAnalysisContent = document.getElementById('imageAnalysisContent');
    
    if (imageAnalysis.success && imageAnalysis.analysis) {
        imageAnalysisCard.style.display = 'block';
        imageAnalysisContent.textContent = imageAnalysis.analysis;
    } else if (imageAnalysis.error) {
        imageAnalysisCard.style.display = 'block';
        imageAnalysisContent.innerHTML = `<em style="color: var(--danger);">Error: ${imageAnalysis.error}</em>`;
    } else {
        imageAnalysisCard.style.display = 'none';
    }
}
