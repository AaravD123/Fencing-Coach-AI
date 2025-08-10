# One Click Start Script for Fencing Coach AI

Write-Host "=== Fencing Coach AI: One Click Setup & Launch ===" -ForegroundColor Cyan

# Step 1: Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python 3.10+ and re-run this script." -ForegroundColor Red
    exit
}

# Step 2: Install required Python packages
Write-Host "Installing required Python packages..." -ForegroundColor Yellow
pip install --upgrade pip
pip install streamlit requests ollama

# Step 3: Check Ollama
$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaInstalled) {
    Write-Host "Ollama not found. Installing..." -ForegroundColor Yellow
    Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" -OutFile "$env:TEMP\OllamaSetup.exe"
    Start-Process "$env:TEMP\OllamaSetup.exe" -Wait
}

# Step 4: Ensure Ollama is running
Write-Host "Starting Ollama server..." -ForegroundColor Yellow
Start-Process ollama serve
Start-Sleep -Seconds 5

# Step 5: Pull the model if not already installed
Write-Host "Ensuring Mistral model is installed..." -ForegroundColor Yellow
ollama pull mistral

# Step 6: Launch Streamlit app
Write-Host "Launching Fencing Coach AI app..." -ForegroundColor Green
streamlit run "$PSScriptRoot\..\app.py"
