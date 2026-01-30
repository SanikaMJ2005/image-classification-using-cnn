// ============================================================================
// DOM Elements
// ============================================================================

const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');
const predictionBox = document.getElementById('predictionBox');
const errorBox = document.getElementById('errorBox');
const errorMessage = document.getElementById('errorMessage');

// ============================================================================
// Event Listeners
// ============================================================================

// Handle upload button click
uploadBtn.addEventListener('click', () => {
    imageInput.click();
});

// Handle file selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

// Drag and drop functionality
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#8b5cf6';
    uploadBox.style.backgroundColor = 'rgba(139, 92, 246, 0.15)';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#6366f1';
    uploadBox.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#6366f1';
    uploadBox.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            handleImageUpload(file);
        } else {
            showError('Please upload an image file');
        }
    }
});

// ============================================================================
// Functions
// ============================================================================

function handleImageUpload(file) {
    // Validate file size
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        showError('File size exceeds 16MB limit');
        return;
    }

    // Read and display image
    const reader = new FileReader();
    reader.onload = (e) => {
        const imageData = e.target.result;
        
        // Show preview
        previewImage.src = imageData;
        previewBox.style.display = 'block';
        errorBox.style.display = 'none';
        
        // Send to server for prediction
        predictImage(file);
    };
    
    reader.readAsDataURL(file);
}

function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    const originalText = uploadBtn.textContent;
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processing...';

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showError(data.error);
        } else {
            displayPrediction(data);
            errorBox.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Failed to process image. Please ensure the model is loaded.');
    })
    .finally(() => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = originalText;
    });
}

function displayPrediction(data) {
    // Update predicted digit
    const predictedDigitElement = document.getElementById('predictedDigit');
    predictedDigitElement.textContent = data.prediction;

    // Update confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercentage = document.getElementById('confidencePercentage');
    
    confidenceFill.style.width = data.confidence + '%';
    confidencePercentage.textContent = data.confidence.toFixed(2) + '%';

    // Display all probabilities
    const probabilitiesContainer = document.getElementById('probabilitiesContainer');
    probabilitiesContainer.innerHTML = '';

    const allPredictions = data.all_predictions;
    
    Object.keys(allPredictions).sort((a, b) => allPredictions[b] - allPredictions[a]).forEach(digit => {
        const probability = allPredictions[digit];
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        // Highlight the predicted digit
        if (parseInt(digit) === data.prediction) {
            item.style.borderLeftColor = '#10b981';
            item.style.backgroundColor = 'rgba(16, 185, 129, 0.05)';
        }
        
        item.innerHTML = `
            <span class="probability-label">Digit ${digit}:</span>
            <span class="probability-value">${probability.toFixed(2)}%</span>
        `;
        probabilitiesContainer.appendChild(item);
    });

    // Show prediction box
    predictionBox.style.display = 'block';

    // Animate confidence bar
    animateProgressBar();
}

function animateProgressBar() {
    const confidenceFill = document.getElementById('confidenceFill');
    const targetWidth = confidenceFill.style.width;
    confidenceFill.style.width = '0';
    
    setTimeout(() => {
        confidenceFill.style.transition = 'width 0.8s ease';
        confidenceFill.style.width = targetWidth;
    }, 10);
}

function showError(message) {
    errorMessage.textContent = message;
    errorBox.style.display = 'block';
    predictionBox.style.display = 'none';
}

// ============================================================================
// Page Load
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Check model status
    checkModelStatus();
    
    // Set up smooth scrolling for navigation links
    setupSmoothScrolling();
});

function checkModelStatus() {
    fetch('/api/model-info')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'loaded') {
                console.warn('Model not loaded. Please train the model first.');
                // Optionally show a warning to the user
            }
        })
        .catch(error => {
            console.error('Error checking model status:', error);
        });
}

function setupSmoothScrolling() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            
            if (href === '#') {
                e.preventDefault();
                return;
            }
            
            const target = document.querySelector(href);
            
            if (target) {
                e.preventDefault();
                // Remove active class from all links
                document.querySelectorAll('.nav-link').forEach(l => {
                    l.classList.remove('active');
                });
                // Add active class to current link
                link.classList.add('active');
                
                // Scroll to target
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// ============================================================================
// Utility Functions
// ============================================================================

// Format confidence score
function formatConfidence(value) {
    return (value * 100).toFixed(2);
}

// Get color based on confidence
function getConfidenceColor(confidence) {
    if (confidence >= 90) return '#10b981'; // green
    if (confidence >= 70) return '#3b82f6'; // blue
    if (confidence >= 50) return '#f59e0b'; // amber
    return '#ef4444'; // red
}
