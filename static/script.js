

class PDFOutlineExtractor {
    constructor() {
        this.currentResults = [];
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkModelStatus();
    }

    bindEvents() {
        
        document.getElementById('training-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadTrainingData();
        });

        document.getElementById('train-btn').addEventListener('click', () => {
            this.trainModel();
        });

        
        document.getElementById('single-process-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.processSinglePDF();
        });

        document.getElementById('batch-process-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.processBatchPDFs();
        });

        
        document.getElementById('download-all-btn').addEventListener('click', () => {
            this.downloadAllResults();
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/model_status');
            const data = await response.json();
            
            if (response.ok) {
                this.updateModelStatus(data);
            } else {
                this.showToast('Error checking model status', 'error');
            }
        } catch (error) {
            this.showToast('Failed to check model status', 'error');
            console.error('Model status check failed:', error);
        }
    }

    updateModelStatus(status) {
        const container = document.getElementById('model-status');
        
        if (status.model_trained) {
            container.innerHTML = `
                <div class="d-flex align-items-center justify-content-between">
                    <div>
                        <span class="model-status-badge status-trained">
                            <i class="fas fa-check-circle"></i>
                            Model Trained
                        </span>
                        <div class="mt-2">
                            <small class="text-muted">
                                Features: ${status.feature_count || 'N/A'} | 
                                Classes: ${status.classes ? status.classes.join(', ') : 'N/A'}
                            </small>
                        </div>
                    </div>
                    <button class="btn btn-outline-primary btn-sm" onclick="extractor.checkModelStatus()">
                        <i class="fas fa-sync-alt"></i>
                        Refresh
                    </button>
                </div>
            `;
            
            
            document.getElementById('process-single-btn').disabled = false;
            document.getElementById('process-batch-btn').disabled = false;
        } else {
            container.innerHTML = `
                <div class="d-flex align-items-center justify-content-between">
                    <div>
                        <span class="model-status-badge status-not-trained">
                            <i class="fas fa-exclamation-triangle"></i>
                            No Model Trained
                        </span>
                        <div class="mt-2">
                            <small class="text-muted">Please train a model first using the training section above</small>
                        </div>
                    </div>
                    <button class="btn btn-outline-primary btn-sm" onclick="extractor.checkModelStatus()">
                        <i class="fas fa-sync-alt"></i>
                        Refresh
                    </button>
                </div>
            `;
            
            
            document.getElementById('process-single-btn').disabled = true;
            document.getElementById('process-batch-btn').disabled = true;
        }
    }

    async uploadTrainingData() {
        const pdfFile = document.getElementById('pdf-file').files[0];
        const jsonFile = document.getElementById('json-file').files[0];
        
        if (!pdfFile || !jsonFile) {
            this.showToast('Please select both PDF and JSON files', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('pdf_file', pdfFile);
        formData.append('json_file', jsonFile);

        const uploadBtn = document.getElementById('upload-btn');
        const originalText = uploadBtn.innerHTML;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
        uploadBtn.disabled = true;

        try {
            const response = await fetch('/upload_training_data', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showToast('Training files uploaded successfully', 'success');
                document.getElementById('train-btn').disabled = false;
                
                
                this.trainingFiles = {
                    pdf_file: data.pdf_file,
                    json_file: data.json_file
                };
            } else {
                this.showToast(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            this.showToast('Upload failed', 'error');
            console.error('Upload error:', error);
        } finally {
            uploadBtn.innerHTML = originalText;
            uploadBtn.disabled = false;
        }
    }

    async trainModel() {
        if (!this.trainingFiles) {
            this.showToast('Please upload training files first', 'error');
            return;
        }

        const trainBtn = document.getElementById('train-btn');
        const originalText = trainBtn.innerHTML;
        const progressDiv = document.getElementById('training-progress');
        
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
        trainBtn.disabled = true;
        progressDiv.style.display = 'block';

        try {
            const response = await fetch('/train_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.trainingFiles)
            });

            const data = await response.json();

            if (response.ok) {
                this.showToast('Model trained successfully!', 'success');
                this.displayTrainingResult(data.model_info);
                this.checkModelStatus(); 
            } else {
                this.showToast(data.error || 'Training failed', 'error');
            }
        } catch (error) {
            this.showToast('Training failed', 'error');
            console.error('Training error:', error);
        } finally {
            trainBtn.innerHTML = originalText;
            trainBtn.disabled = false;
            progressDiv.style.display = 'none';
        }
    }

    displayTrainingResult(modelInfo) {
        const resultDiv = document.getElementById('training-result');
        
        resultDiv.innerHTML = `
            <div class="alert alert-success mt-3">
                <h6><i class="fas fa-chart-line"></i> Training Results</h6>
                <div class="row">
                    <div class="col-md-6">
                        <strong>Accuracy:</strong> ${(modelInfo.accuracy * 100).toFixed(1)}%<br>
                        <strong>Samples:</strong> ${modelInfo.num_samples}<br>
                        <strong>Features:</strong> ${modelInfo.num_features}
                    </div>
                    <div class="col-md-6">
                        <strong>Classes:</strong> ${modelInfo.classes.join(', ')}<br>
                        <strong>Label Distribution:</strong><br>
                        ${Object.entries(modelInfo.label_distribution)
                            .map(([label, count]) => `${label}: ${count}`)
                            .join(', ')}
                    </div>
                </div>
            </div>
        `;
    }

    async processSinglePDF() {
        const pdfFile = document.getElementById('single-pdf').files[0];
        
        if (!pdfFile) {
            this.showToast('Please select a PDF file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('pdf_file', pdfFile);

        const processBtn = document.getElementById('process-single-btn');
        const originalText = processBtn.innerHTML;
        const progressDiv = document.getElementById('processing-progress');
        
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        processBtn.disabled = true;
        progressDiv.style.display = 'block';

        try {
            const response = await fetch('/process_pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showToast('PDF processed successfully', 'success');
                data.filename = pdfFile.name;
                this.currentResults = [data];
                this.displayResults(this.currentResults);
            } else {
                this.showToast(data.error || 'Processing failed', 'error');
            }
        } catch (error) {
            this.showToast('Processing failed', 'error');
            console.error('Processing error:', error);
        } finally {
            processBtn.innerHTML = originalText;
            processBtn.disabled = false;
            progressDiv.style.display = 'none';
        }
    }

    async processBatchPDFs() {
        const pdfFiles = document.getElementById('batch-pdfs').files;
        
        if (pdfFiles.length === 0) {
            this.showToast('Please select PDF files', 'error');
            return;
        }

        const formData = new FormData();
        for (let file of pdfFiles) {
            formData.append('pdf_files', file);
        }

        const processBtn = document.getElementById('process-batch-btn');
        const originalText = processBtn.innerHTML;
        const progressDiv = document.getElementById('processing-progress');
        
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        processBtn.disabled = true;
        progressDiv.style.display = 'block';

        try {
            const response = await fetch('/batch_process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showToast(`Processed ${data.results.length} files`, 'success');
                this.currentResults = data.results;
                this.displayResults(this.currentResults);
            } else {
                this.showToast(data.error || 'Batch processing failed', 'error');
            }
        } catch (error) {
            this.showToast('Batch processing failed', 'error');
            console.error('Batch processing error:', error);
        } finally {
            processBtn.innerHTML = originalText;
            processBtn.disabled = false;
            progressDiv.style.display = 'none';
        }
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        const downloadBtn = document.getElementById('download-all-btn');
        
        if (results.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No results to display.</p>';
            downloadBtn.style.display = 'none';
            return;
        }

        let html = '';
        
        results.forEach((result, index) => {
            if (result.error) {
                html += `
                    <div class="result-item">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="text-danger">
                                <i class="fas fa-exclamation-triangle"></i>
                                ${result.filename || `File ${index + 1}`}
                            </h6>
                        </div>
                        <div class="alert alert-danger mt-2">
                            Error: ${result.error}
                        </div>
                    </div>
                `;
            } else {
                html += `
                    <div class="result-item">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6>
                                <i class="fas fa-file-pdf text-danger"></i>
                                ${result.filename || `File ${index + 1}`}
                            </h6>
                            <button class="btn btn-outline-primary btn-sm" onclick="extractor.downloadSingleResult(${index})">
                                <i class="fas fa-download"></i>
                                Download JSON
                            </button>
                        </div>
                        
                        <div class="mt-3">
                            <h6 class="text-primary">
                                <i class="fas fa-heading"></i>
                                Title: ${result.title}
                            </h6>
                        </div>
                        
                        <div class="mt-3">
                            <h6>
                                <i class="fas fa-list"></i>
                                Outline (${result.outline.length} items):
                            </h6>
                            ${result.outline.length > 0 ? 
                                result.outline.map(item => `
                                    <div class="outline-item level-${item.level.toLowerCase()}">
                                        <span class="badge bg-secondary me-2">${item.level}</span>
                                        <span class="badge bg-info me-2">Page ${item.page}</span>
                                        ${item.text}
                                    </div>
                                `).join('') :
                                '<p class="text-muted">No outline items found</p>'
                            }
                        </div>
                        
                        ${result.metadata ? `
                            <div class="metadata-info">
                                <h6 class="mb-2">
                                    <i class="fas fa-info-circle"></i>
                                    Metadata
                                </h6>
                                <div class="metadata-item">
                                    <i class="fas fa-file"></i>
                                    Pages: ${result.metadata.total_pages}
                                </div>
                                <div class="metadata-item">
                                    <i class="fas fa-text"></i>
                                    Text Lines: ${result.metadata.total_text_lines}
                                </div>
                                <div class="metadata-item">
                                    <i class="fas fa-list"></i>
                                    Outline Items: ${result.metadata.outline_items}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
            }
        });
        
        container.innerHTML = html;
        downloadBtn.style.display = results.length > 0 ? 'inline-block' : 'none';
    }

    downloadSingleResult(index) {
        const result = this.currentResults[index];
        if (!result) return;

        const filename = (result.filename || `result_${index + 1}.pdf`).replace('.pdf', '.json');
        const cleanResult = { ...result };
        delete cleanResult.filename;

        const blob = new Blob([JSON.stringify(cleanResult, null, 4)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async downloadAllResults() {
        if (this.currentResults.length === 0) {
            this.showToast('No results to download', 'error');
            return;
        }

        try {
            const response = await fetch('/download_results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ results: this.currentResults })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'pdf_outline_results.zip';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showToast('Results downloaded successfully', 'success');
            } else {
                const data = await response.json();
                this.showToast(data.error || 'Download failed', 'error');
            }
        } catch (error) {
            this.showToast('Download failed', 'error');
            console.error('Download error:', error);
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toastId = 'toast-' + Date.now();
        
        const bgClass = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type] || 'bg-info';

        const iconClass = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        }[type] || 'fas fa-info-circle';

        const toastHtml = `
            <div id="${toastId}" class="toast ${bgClass} text-white" role="alert">
                <div class="toast-header ${bgClass} text-white border-0">
                    <i class="${iconClass} me-2"></i>
                    <strong class="me-auto">PDF Outline Extractor</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: type === 'error' ? 5000 : 3000
        });
        
        toast.show();
        

        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
}


let extractor;
document.addEventListener('DOMContentLoaded', () => {
    extractor = new PDFOutlineExtractor();
});
