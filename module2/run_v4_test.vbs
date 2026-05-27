Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "d:\pythonProject"
WshShell.Run "py module2\source_isolated_arbitration.py --dataset dataset_amp_8.jsonl --tag v4_test --limit 3", 0, False
