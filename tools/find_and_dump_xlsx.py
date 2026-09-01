"""在 Temp 目录里按通配符定位协议 xlsx 并导出为文本。

绕过 PowerShell 中文路径编码问题：全程在 Python 内处理路径。
"""

from __future__ import annotations

import glob
import os
import sys

import openpyxl


def cell_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.replace("\r\n", "\n").replace("\n", " ⏎ ")
    return str(v)


def dump(src: str, dst: str) -> None:
    wb = openpyxl.load_workbook(src, data_only=True)
    lines: list[str] = []
    lines.append(f"FILE: {src}")
    lines.append(f"SHEETS({len(wb.sheetnames)}): {wb.sheetnames}")

    for ws in wb.worksheets:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"### SHEET: {ws.title}  max_row={ws.max_row} max_col={ws.max_column}")
        lines.append("=" * 78)
        for row in ws.iter_rows():
            vals = [cell_text(c.value).strip() for c in row]
            if not any(vals):
                continue
            lines.append(f"R{row[0].row:>4}| " + " | ".join(vals))

    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SHEETS:", wb.sheetnames)
    print("WRITTEN:", dst)


def main() -> int:
    dst = sys.argv[1] if len(sys.argv) > 1 else r"f:\2026\IrisLoop\tools\ble_protocol_dump.txt"

    # 先在 Temp 下找，再退到常见下载目录
    temp = os.environ.get("TEMP", r"C:\Windows\Temp")
    roots = [
        os.path.join(temp, "codebuddy-dropped-files"),
        temp,
        os.path.join(os.path.expanduser("~"), "Downloads"),
    ]

    found: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in ("*蓝牙*.xlsx", "*协议*.xlsx", "*Iris*.xlsx"):
            found.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

    # 去重
    seen = set()
    uniq = []
    for p in found:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    print("CANDIDATES:")
    for p in uniq:
        print("  ", p, os.path.getsize(p))

    if not uniq:
        print("[error] 未找到协议 xlsx")
        return 1

    # 优先名字里带 协议/蓝牙 的
    target = None
    for p in uniq:
        b = os.path.basename(p)
        if "协议" in b or "蓝牙" in b:
            target = p
            break
    target = target or uniq[0]

    print("\nUSING:", target)
    dump(target, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
