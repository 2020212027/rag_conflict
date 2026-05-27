Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d d:\pythonProject && .venv\Scripts\python.exe time_benchmark.py > time_benchmark_output.txt 2>&1", 0, False
