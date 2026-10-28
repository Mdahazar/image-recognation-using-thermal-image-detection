// Global variables
let currentFile = null;
let currentAnalysis = null; // ADDED: Store analysis data for PDF download

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeHomePage();
    initializeAnalysisPage();
    setupEventListeners();
});

// Home page initialization
function initializeHomePage() {
    if (document.querySelector('.hero-section')) {
        loadStatistics();
        startCounterAnimations();
    }
}

// Analysis page initialization
function initializeAnalysisPage() {
    if (document.getElementById('uploadForm')) {
        setupDragAndDrop();
        setupFileInput();
        setupDownloadButton(); // ADDED: Setup download button
    }
}

// Setup event listeners
function setupEventListeners() {
    // Global escape key handler
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModals();
        }
    });
}

// ADDED: Setup download button functionality
function setupDownloadButton() {
    const downloadButton = document.querySelector('.action-button.primary');
    if (downloadButton) {
        downloadButton.addEventListener('click', downloadReport);
    }
}

// ADDED: Download PDF report
async function downloadReport() {
    if (!currentAnalysis) {
        showNotification('No analysis data available for download', 'error');
        return;
    }
    
    const downloadButton = document.querySelector('.action-button.primary');
    const originalText = downloadButton.innerHTML;
    
    try {
        // Show loading state
        downloadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
        downloadButton.disabled = true;
        
        const response = await fetch('/download_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis: currentAnalysis.analysis,
                original_image: currentAnalysis.images.original,
                thermal_image: currentAnalysis.images.thermal_jet
            })
        });
        
        if (response.ok) {
            // Create blob from response and trigger download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `ThermoScan_Report_${currentAnalysis.analysis.analysis_id}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showNotification('Report downloaded successfully!', 'success');
        } else {
            const error = await response.json();
            showNotification(error.message || 'Error downloading report', 'error');
        }
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Error downloading report. Please try again.', 'error');
    } finally {
        // Restore button state
        downloadButton.innerHTML = originalText;
        downloadButton.disabled = false;
    }
}

// Statistics loading for home page
async function loadStatistics() {
    try {
        const response = await fetch('/api/analysis_stats');
        const stats = await response.json();
        
        animateCounter('totalAnalyses', stats.total_analyses);
        animateCounter('accuracyRate', stats.accuracy_rate, true);
        animateCounter('processingTime', stats.avg_processing_time, false, true);
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Counter animation
function animateCounter(elementId, targetValue, isPercentage = false, isTime = false) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    let current = 0;
    const target = isPercentage ? parseInt(targetValue) : 
                   isTime ? parseFloat(targetValue) : 
                   parseInt(targetValue);
    const increment = target / 50;
    const duration = 1500;
    const stepTime = duration / 50;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        
        if (isPercentage) {
            element.textContent = Math.round(current) + '%';
        } else if (isTime) {
            element.textContent = current.toFixed(1) + 's';
        } else {
            element.textContent = Math.round(current).toLocaleString();
        }
    }, stepTime);
}

// Drag and drop functionality
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) return;
    
    // Drag over event
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    // Drag leave event
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('drag-over');
    });
    
    // Drop event
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
}

// File input setup - COMPLETELY REWRITTEN
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    const browseButton = document.getElementById('browseButton');
    const uploadArea = document.getElementById('uploadArea');
    
    if (!fileInput || !browseButton) return;
    
    // ONLY browse button triggers file input
    browseButton.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Browse button clicked');
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', function(e) {
        console.log('File input changed');
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
    
    // PREVENT upload area from triggering file input
    uploadArea.addEventListener('click', function(e) {
        // Only prevent if click is on the upload area itself, not on buttons inside
        if (e.target === uploadArea) {
            e.preventDefault();
            e.stopPropagation();
        }
    });
}

// Handle file selection
function handleFileSelection(file) {
    if (!file) return;
    
    console.log('File selected:', file.name);
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please select a valid image file (PNG, JPG, JPEG, BMP, TIFF)', 'error');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size must be less than 16MB', 'error');
        return;
    }
    
    currentFile = file;
    displayFilePreview(file);
    enableAnalyzeButton();
}

// Display file preview
function displayFilePreview(file) {
    const preview = document.getElementById('filePreview');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    // Create object URL for preview
    const objectUrl = URL.createObjectURL(file);
    
    previewImage.src = objectUrl;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    preview.style.display = 'block';
    
    // Scroll to preview
    preview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Remove file
function removeFile() {
    const preview = document.getElementById('filePreview');
    const fileInput = document.getElementById('fileInput');
    
    preview.style.display = 'none';
    fileInput.value = '';
    currentFile = null;
    currentAnalysis = null; // ADDED: Clear analysis data
    disableAnalyzeButton();
}

// Enable analyze button
function enableAnalyzeButton() {
    const analyzeButton = document.getElementById('analyzeButton');
    analyzeButton.disabled = false;
}

// Disable analyze button
function disableAnalyzeButton() {
    const analyzeButton = document.getElementById('analyzeButton');
    analyzeButton.disabled = true;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Form submission
document.getElementById('uploadForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    if (!currentFile) {
        showNotification('Please select a file first', 'error');
        return;
    }
    
    await uploadAndAnalyze();
});

// Upload and analyze
async function uploadAndAnalyze() {
    const formData = new FormData();
    formData.append('file', currentFile);
    
    const analyzeButton = document.getElementById('analyzeButton');
    const buttonLoading = document.getElementById('buttonLoading');
    
    // Show loading state
    analyzeButton.disabled = true;
    buttonLoading.style.display = 'block';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification(result.message, 'success');
            displayAnalysisResults(result);
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('An error occurred during upload. Please try again.', 'error');
    } finally {
        // Hide loading state
        analyzeButton.disabled = false;
        buttonLoading.style.display = 'none';
    }
}

// Display analysis results
function displayAnalysisResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const analysis = result.analysis;
    const images = result.images;
    
    // ADDED: Store analysis data for PDF download
    currentAnalysis = result;
    
    // Update analysis metadata
    document.getElementById('analysisTime').textContent = analysis.analysis_time;
    document.getElementById('analysisId').textContent = analysis.analysis_id;
    
    // Update temperature data
    document.getElementById('maxTemp').textContent = analysis.max_temperature;
    document.getElementById('minTemp').textContent = analysis.min_temperature;
    document.getElementById('avgTemp').textContent = analysis.avg_temperature;
    
    // Update detection data
    document.getElementById('hotSpots').textContent = analysis.hot_spots;
    document.getElementById('criticalZones').textContent = analysis.critical_zones;
    document.getElementById('confidenceLevel').textContent = analysis.confidence_level;
    
    // NEW: Update object detection data
    document.getElementById('detectedObject').textContent = analysis.detected_object || 'Unknown';
    document.getElementById('objectConfidence').textContent = analysis.object_confidence || '0%';
    
    // Update image information
    document.getElementById('imageSize').textContent = analysis.image_size;
    document.getElementById('imageResolution').textContent = analysis.resolution;
    document.getElementById('thermalVariance').textContent = analysis.thermal_variance;
    
    // Update images
    document.getElementById('originalImage').src = images.original;
    document.getElementById('thermalImage').src = images.thermal_jet;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);
}

// Reset analysis
function resetAnalysis() {
    const resultsSection = document.getElementById('resultsSection');
    const filePreview = document.getElementById('filePreview');
    const fileInput = document.getElementById('fileInput');
    
    resultsSection.style.display = 'none';
    filePreview.style.display = 'none';
    fileInput.value = '';
    currentFile = null;
    currentAnalysis = null; // ADDED: Clear analysis data
    disableAnalyzeButton();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    
    if (!notification) return;
    
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 5000);
}

// Close modals (placeholder for future features)
function closeModals() {
    // Implement modal closing logic if needed
}

// Start counter animations
function startCounterAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                loadStatistics();
                observer.unobserve(entry.target);
            }
        });
    });
    
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        observer.observe(heroSection);
    }
}