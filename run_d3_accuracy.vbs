Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d d:\pythonProject && .venv\Scripts\python.exe exp_d3_accuracy.py > d3_accuracy.log 2>&1", 0, False
