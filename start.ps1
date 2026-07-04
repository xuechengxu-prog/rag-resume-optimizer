# 企业级简历优化RAG系统 - PowerShell启动脚本
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  企业级简历优化RAG系统 - 启动脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[错误] 未检测到Python，请先安装Python 3.10+" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

# 创建虚拟环境
if (-not (Test-Path "venv")) {
    Write-Host "[1/4] 创建虚拟环境..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 虚拟环境创建失败" -ForegroundColor Red
        Read-Host "按任意键退出"
        exit 1
    }
}

# 激活虚拟环境
Write-Host "[2/4] 激活虚拟环境..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# 安装依赖
if (-not (Test-Path "venv\installed.flag")) {
    Write-Host "[3/4] 安装依赖（首次运行需要几分钟）..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 依赖安装失败" -ForegroundColor Red
        Read-Host "按任意键退出"
        exit 1
    }
    "installed" | Out-File -FilePath "venv\installed.flag" -Encoding utf8
} else {
    Write-Host "[3/4] 依赖已安装，跳过..." -ForegroundColor Green
}

# 检查环境变量
if (-not $env:DASHSCOPE_API_KEY) {
    Write-Host "[警告] 未检测到 DASHSCOPE_API_KEY 环境变量" -ForegroundColor Yellow
    Write-Host "         请在系统环境变量中配置，或在项目目录创建 .env 文件" -ForegroundColor Yellow
    Write-Host "         格式: DASHSCOPE_API_KEY=your_api_key_here" -ForegroundColor Yellow
    Write-Host ""
}

# 启动服务
Write-Host "[4/4] 启动服务..." -ForegroundColor Yellow
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  服务地址: http://localhost:8000" -ForegroundColor Green
Write-Host "  API文档:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "按 Ctrl+C 停止服务" -ForegroundColor Gray
Write-Host ""

python api.py

Read-Host "按任意键退出"
