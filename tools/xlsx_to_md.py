"""把 doc/Iris Green蓝牙通信协议_20260324_V1.9.xlsx 转成 Markdown 保存到 doc/。

用法: python tools/xlsx_to_md.py
"""

from __future__ import annotations

import os

import openpyxl

SRC = r"doc\Iris Green蓝牙通信协议_20260324_V1.9.xlsx"
DST = r"doc\Iris Green蓝牙通信协议_20260324_V1.9.md"


def cell_str(v) -> str:
    if v is None:
        return ""
    s = str(v).replace("\r\n", "\n").replace("\r", "\n")
    # 单元格内换行用 <br>，竖线转义，避免破坏 md 表格
    return s.replace("|", "\\|").replace("\n", "<br>").strip()


def sheet_to_md(ws, out: list[str]) -> None:
    # 合并单元格值填充（只取左上角值）
    merged: dict[tuple[int, int], object] = {}
    for rng in ws.merged_cells.ranges:
        v = ws.cell(row=rng.min_row, column=rng.min_col).value
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                merged[(r, c)] = v

    def get(r: int, c: int) -> str:
        v = ws.cell(row=r, column=c).value
        if v is None and (r, c) in merged:
            v = merged[(r, c)]
        return cell_str(v)

    def dedup(row: list[str]) -> list[str]:
        """合并单元格横向填充会产生重复列，同一行内相同值只保留首次出现。"""
        seen: set[str] = set()
        out: list[str] = []
        for v in row:
            if v and v in seen:
                out.append("")
            else:
                if v:
                    seen.add(v)
                out.append(v)
        return out

    max_r, max_c = ws.max_row, ws.max_column
    # 修剪全空尾列
    while max_c > 1 and all(not get(r, max_c) for r in range(1, max_r + 1)):
        max_c -= 1

    for r in range(1, max_r + 1):
        row = dedup([get(r, c) for c in range(1, max_c + 1)])
        # 去掉行首/行尾连续空列
        while row and not row[0]:
            row.pop(0)
        while row and not row[-1]:
            row.pop()
        if not row:
            continue
        # 单格非空 -> 段落文本；多格 -> 表格行
        nonempty = [i for i, v in enumerate(row) if v]
        if len(nonempty) <= 1:
            v = row[nonempty[0]] if nonempty else ""
            if v.isdigit():  # 纯数字残留行号，丢弃
                continue
            out.append(v)
            out.append("")
        else:
            out.append("| " + " | ".join(row) + " |")
            # 在连续表格块首行后补分隔行
            if len(out) < 2 or not out[-2].startswith("|"):
                ncols = len(row)
                out.insert(len(out) - 1, "|" + "---|" * ncols)


def main() -> None:
    wb = openpyxl.load_workbook(SRC, data_only=True)
    out: list[str] = [
        "# Iris Green 蓝牙通信协议 V1.9（2026-03-24）",
        "",
        f"> 由 `{os.path.basename(SRC)}` 自动导出（tools/xlsx_to_md.py），仅作查阅，",
        "> 协议变更请改 xlsx 源文件后重新导出。",
        "",
    ]
    for name in wb.sheetnames:
        ws = wb[name]
        out.append(f"## Sheet: {name}")
        out.append("")
        sheet_to_md(ws, out)
        out.append("")

    text = "\n".join(out)
    # 清理 3+ 连续空行
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    with open(DST, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"written {DST}  {len(text)} chars")


if __name__ == "__main__":
    main()
