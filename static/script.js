// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');

// File input change event
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop functionality
uploadBox.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
    
    const file = e.dataTransfer.files[0];
    
    if (file && isValidImageFile(file)) {
        // Update file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        handleFileSelect(file);
    } else {
        alert('Please upload a valid image file (PNG, JPG, JPEG)');
    }
});

// Handle file selection
function handleFileSelect(file) {
    // Validate file
    if (!isValidImageFile(file)) {
        alert('Invalid file type. Please upload PNG, JPG, or JPEG.');
        return;
    }
    
    // Check file size (16MB max)
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        alert('File size exceeds 16MB. Please upload a smaller image.');
        return;
    }
    
    // Read and display image
    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadBox.style.display = 'none';
        detectBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

// Validate image file type
function isValidImageFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    return validTypes.includes(file.type);
}

// Remove image button
removeBtn.addEventListener('click', function() {
    // Clear file input
    fileInput.value = '';
    
    // Hide preview
    previewContainer.style.display = 'none';
    uploadBox.style.display = 'block';
    
    // Clear preview
    imagePreview.src = '';
    
    // Disable detect button
    detectBtn.disabled = true;
});

// Form submission handling
const uploadForm = document.getElementById('uploadForm');

uploadForm.addEventListener('submit', function(e) {
    // Show loading state
    detectBtn.textContent = '🔍 Detecting...';
    detectBtn.disabled = true;
    
    // Form will submit normally to Flask backend
});

// Auto-hide flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            
            setTimeout(function() {
                alert.remove();
            }, 500);
        }, 5000);
    });
});

// Smooth scroll to results if present
window.addEventListener('load', function() {
    const resultSection = document.querySelector('.result-section');
    
    if (resultSection) {
        setTimeout(function() {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
});
const bar = document.querySelector(".confidence-fill");
if (bar) {
    bar.style.width = bar.dataset.confidence + "%";
}
