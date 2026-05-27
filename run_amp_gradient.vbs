Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d d:\pythonProject && .venv\Scripts\python.exe exp_amp_gradient.py amp2 > layered_amp2.log 2>&1 && .venv\Scripts\python.exe exp_amp_gradient.py amp4 > layered_amp4.log 2>&1", 0, False
