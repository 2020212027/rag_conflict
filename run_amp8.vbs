Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d d:\pythonProject & py step_4_layered_e2e.py amp8 > layered_amp8.log 2>&1", 0, False
