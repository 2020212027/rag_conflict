Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "d:\pythonProject"
WshShell.Run "py step_4_layered_e2e.py clean", 0, False
