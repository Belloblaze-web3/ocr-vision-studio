const $ = (selector) => document.querySelector(selector);
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const browseButton = $('#browse-button');
const runButton = $('#run-button');
const fileCard = $('#file-card');
const sourcePreview = $('#source-preview');
const errorMessage = $('#error-message');
const processingState = $('#processing-state');
let selectedFile = null;
let latestResult = null;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove('is-hidden');
}

function clearError() { errorMessage.classList.add('is-hidden'); }

function selectFile(file) {
  clearError();
  if (!file) return;
  if (!file.type.startsWith('image/')) return showError('Please choose an image file.');
  if (file.size > 10 * 1024 * 1024) return showError('Images must be 10 MB or smaller.');
  selectedFile = file;
  sourcePreview.src = URL.createObjectURL(file);
  $('#file-name').textContent = file.name;
  $('#file-meta').textContent = `${file.type.split('/')[1].toUpperCase()} · ${formatBytes(file.size)}`;
  fileCard.classList.remove('is-hidden');
  dropZone.classList.add('is-hidden');
  runButton.disabled = false;
  $('#result-status').textContent = 'READY TO RUN';
}

function resetAll() {
  selectedFile = null;
  latestResult = null;
  fileInput.value = '';
  sourcePreview.removeAttribute('src');
  fileCard.classList.add('is-hidden');
  dropZone.classList.remove('is-hidden');
  runButton.disabled = true;
  $('#evidence-empty').classList.remove('is-hidden');
  $('#evidence-result').classList.add('is-hidden');
  $('#results-section').classList.add('is-hidden');
  $('#result-status').textContent = 'AWAITING IMAGE';
  clearError();
}

function setProcessing(active) {
  runButton.disabled = active || !selectedFile;
  processingState.classList.toggle('is-hidden', !active);
  if (active) $('#result-status').textContent = 'PROCESSING';
}

function renderDetectionRows(result) {
  const body = $('#detection-body');
  const rows = [...result.accepted.map((item) => ({ ...item, gate: 'ACCEPTED' })), ...result.review.map((item) => ({ ...item, gate: 'REVIEW' }))];
  body.innerHTML = rows.map((item) => {
    const box = item.box;
    return `<tr><td><strong>${escapeHtml(item.text)}</strong></td><td><span class="confidence">${item.confidence.toFixed(1)}%</span></td><td><span class="box-coords">x${box.left} · y${box.top} · ${box.width}×${box.height}</span></td><td><span class="gate ${item.gate.toLowerCase()}">${item.gate}</span></td></tr>`;
  }).join('') || '<tr><td colspan="4" class="box-coords">No text tokens detected.</td></tr>';
}

function renderResult(result) {
  latestResult = result;
  $('#evidence-empty').classList.add('is-hidden');
  $('#evidence-result').classList.remove('is-hidden');
  $('#annotated-image').src = result.annotated_image;
  $('#image-size').textContent = `${result.image_size.width} × ${result.image_size.height}px`;
  $('#result-status').textContent = 'ANALYSIS COMPLETE';
  $('#results-section').classList.remove('is-hidden');
  $('#accepted-count').textContent = result.accepted_count;
  $('#review-count').textContent = result.review_count;
  $('#average-confidence').textContent = `${result.average_confidence}%`;
  $('#processing-ms').textContent = `${result.processing_ms}ms`;
  $('#recognized-text').textContent = result.recognized_text || 'No accepted text detected.';
  $('#engine-label').textContent = 'TESSERACT OCR';
  $('#total-count').textContent = result.total_count;
  renderDetectionRows(result);
  $('#results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[character]));
}

async function analyze() {
  if (!selectedFile) return;
  clearError();
  setProcessing(true);
  const formData = new FormData();
  formData.append('image', selectedFile);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Recognition failed.');
    renderResult(result);
  } catch (error) {
    showError(error.message);
    $('#result-status').textContent = 'ANALYSIS FAILED';
  } finally {
    setProcessing(false);
  }
}

function download(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  URL.revokeObjectURL(url);
}

browseButton.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (event) => { if (event.target !== browseButton) fileInput.click(); });
dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', (event) => selectFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
runButton.addEventListener('click', analyze);
$('#remove-file').addEventListener('click', resetAll);
$('#download-json').addEventListener('click', () => { if (latestResult) download('ocr-recognition.json', JSON.stringify({ ...latestResult, annotated_image: undefined }, null, 2), 'application/json'); });
$('#download-text').addEventListener('click', async () => { if (!latestResult) return; const text = latestResult.recognized_text || ''; try { await navigator.clipboard.writeText(text); $('#download-text').innerHTML = 'Copied <span>✓</span>'; setTimeout(() => { $('#download-text').innerHTML = 'Copy text <span>↗</span>'; }, 1400); } catch { download('recognized-text.txt', text, 'text/plain'); } });
