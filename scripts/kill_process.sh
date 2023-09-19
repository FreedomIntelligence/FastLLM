# 终止进程
# pkill -f "python -u src/gpt4_eval.py"
pkill -f "python"

if [ $? -eq 0 ]; then
  echo "Process terminated."
else
  echo "No process found."
fi