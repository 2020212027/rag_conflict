Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d d:\pythonProject && .venv\Scripts\python.exe module3_faitheval\run_faitheval.py 500 > module3_faitheval\faitheval_run.log 2>&1", 0, False
