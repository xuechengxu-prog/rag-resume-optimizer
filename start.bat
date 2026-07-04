@echo off
chcp 65001 >nul
echo ==========================================
echo   企业级简历优化RAG系统 - 启动脚本
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.10+
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist "venv" (
    echo [1/4] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
echo [2/4] 激活虚拟环境...
call venv\Scripts\activate.bat

REM 检查依赖是否已安装
if not exist "venv\installed.flag" (
    echo [3/4] 安装依赖（首次运行需要几分钟）...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
    echo installed > venv\installed.flag
) else (
    echo [3/4] 依赖已安装，跳过...
)

REM 检查环境变量
if "%DASHSCOPE_API_KEY%"=="" (
    echo [警告] 未检测到 DASHSCOPE_API_KEY 环境变量
    echo          请在系统环境变量中配置，或在项目目录创建 .env 文件
    echo          格式: DASHSCOPE_API_KEY=your_api_key_here
    echo.
)

REM 启动服务
echo [4/4] 启动服务...
echo.
echo ==========================================
echo   服务地址: http://localhost:8000
echo   API文档:  http://localhost:8000/docs
echo ==========================================
echo.
echo 按 Ctrl+C 停止服务
echo.

python api.py

pause
