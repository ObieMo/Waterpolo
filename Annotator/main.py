import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import torch
from PIL import Image

VIDEO_PATH = r"match.mp4"
EXCEL_PATH = r"match.xlsx"
SHEET_NAME = "Event_Log_For_Annotation"
ANNOTATIONS_PATH = "annotations.json"
STATE_PATH = "scan_state.json"
DEBUG_DIR = "debug_ocr"

SAMPLE_EVERY_SEC = 1
OCR_CHECKS_PER_SEC = 5
MIN_SAME_READS_FOR_MATCH = 3
PRINT_EVERY = 10
PREVIEW_WIDTH = 1280

CLOCK_MMSS_RE = re.compile(r"(\d{1,2})\s*:\s*([0-5]\d)")
CLOCK_SST_RE = re.compile(r"\b([0-5]?\d)\s*[.,]\s*(\d)\b")


def parse_clock(text: str):
    if not text:
        return None
    m = CLOCK_MMSS_RE.search(text)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if mm > 12:
            return None
        return mm * 60 + ss

    # Last-minute clocks can appear as SS.t (e.g. 21.1).
    d = CLOCK_SST_RE.search(text)
    if d:
        ss = int(d.group(1))
        if 0 <= ss <= 59:
            return ss

    return None


def format_hhmmss(total_seconds):
    s = int(total_seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def parse_hhmmss_to_sec(text):
    raw = text.strip()
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("Expected HH:MM:SS")
    hh, mm, ss = [int(p) for p in parts]
    if hh < 0 or mm < 0 or mm > 59 or ss < 0 or ss > 59:
        raise ValueError("Invalid HH:MM:SS")
    return hh * 3600 + mm * 60 + ss


def input_quarter_windows():
    print("Enter quarter boundaries in video time HH:MM:SS.")
    labels = [
        "Q1 start", "Q1 end", "Q2 start", "Q2 end",
        "Q3 start", "Q3 end", "Q4 start", "Q4 end",
    ]
    values = []
    for label in labels:
        while True:
            try:
                values.append(parse_hhmmss_to_sec(input(f"{label}: ")))
                break
            except Exception:
                print("Invalid format. Use HH:MM:SS, e.g. 00:07:25")

    windows = [
        (1, values[0], values[1]),
        (2, values[2], values[3]),
        (3, values[4], values[5]),
        (4, values[6], values[7]),
    ]
    for p, start, end in windows:
        if end < start:
            raise ValueError(f"Q{p} end must be >= start.")
    return windows


def build_sampling_plan(windows):
    plan = []
    for period, start_sec, end_sec in windows:
        for sec in range(start_sec, end_sec + 1, SAMPLE_EVERY_SEC):
            plan.append((period, sec))
    return plan


def save_json_atomic(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json_list(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def load_state(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_state(path, state):
    save_json_atomic(path, state)


def column_letters_to_index(cell_ref: str):
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheet_rows(xlsx_path, sheet_name):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    office_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with zipfile.ZipFile(xlsx_path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{rel_ns}Relationship")}

        target = None
        for sheet in wb.find("a:sheets", ns).findall("a:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib[office_rel]
                target = rel_map[rid]
                break
        if target is None:
            raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        elif target.startswith("xl/"):
            sheet_path = target
        else:
            sheet_path = "xl/" + target

        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        def cell_value(cell):
            cell_type = cell.attrib.get("t")
            v = cell.find("a:v", ns)
            if v is None:
                inline = cell.find("a:is", ns)
                if inline is None:
                    return ""
                return "".join(t.text or "" for t in inline.findall(".//a:t", ns))
            raw = v.text or ""
            if cell_type == "s":
                try:
                    return shared[int(raw)]
                except Exception:
                    return raw
            return raw

        sheet_xml = ET.fromstring(z.read(sheet_path))
        raw_rows = []
        for row in sheet_xml.findall(".//a:sheetData/a:row", ns):
            values = {}
            max_col = -1
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col_idx = column_letters_to_index(ref) if ref else (max_col + 1)
                values[col_idx] = cell_value(cell)
                if col_idx > max_col:
                    max_col = col_idx
            if max_col < 0:
                raw_rows.append([])
            else:
                dense = [values.get(i, "") for i in range(max_col + 1)]
                raw_rows.append(dense)
        return raw_rows


def load_events_from_excel(xlsx_path, sheet_name):
    rows = read_xlsx_sheet_rows(xlsx_path, sheet_name)
    if not rows:
        return []

    headers = [h.strip() for h in rows[0]]
    idx = {name: i for i, name in enumerate(headers)}

    required = ["row_id", "period", "clock"]
    for key in required:
        if key not in idx:
            raise ValueError(f"Missing required column '{key}' in sheet '{sheet_name}'")

    events = []
    for row in rows[1:]:
        row_id_raw = row[idx["row_id"]] if idx["row_id"] < len(row) else ""
        period_raw = row[idx["period"]] if idx["period"] < len(row) else ""
        clock_raw = row[idx["clock"]] if idx["clock"] < len(row) else ""
        if row_id_raw == "" or period_raw == "" or clock_raw == "":
            continue

        clock_sec = parse_clock(str(clock_raw))
        if clock_sec is None:
            continue

        try:
            row_id = int(float(str(row_id_raw)))
            period = int(float(str(period_raw)))
        except Exception:
            continue

        event = {
            "row_id": row_id,
            "period": period,
            "clock": str(clock_raw),
            "clock_sec": clock_sec,
            "code": row[idx["code"]] if "code" in idx and idx["code"] < len(row) else "",
            "suggested_label": row[idx["suggested_label"]] if "suggested_label" in idx and idx["suggested_label"] < len(row) else "",
            "team": row[idx["team"]] if "team" in idx and idx["team"] < len(row) else "",
            "player": row[idx["player"]] if "player" in idx and idx["player"] < len(row) else "",
            "cap": row[idx["cap"]] if "cap" in idx and idx["cap"] < len(row) else "",
        }
        events.append(event)

    events.sort(key=lambda e: e["row_id"])
    return events


def pick_roi_on_frame(frame):
    h, w = frame.shape[:2]
    scale = PREVIEW_WIDTH / w if w > PREVIEW_WIDTH else 1.0
    preview = cv2.resize(frame, (int(w * scale), int(h * scale)))
    win = "Select CLOCK ROI (Enter/Space confirm, C cancel)"
    roi_preview = cv2.selectROI(win, preview, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, rw, rh = roi_preview
    if rw == 0 or rh == 0:
        return None
    return (int(x / scale), int(y / scale), int(rw / scale), int(rh / scale))


def load_parseq_tiny():
    model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=True).eval()
    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    return model, transform


def parseq_predict_clock(model, transform, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, ""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = logits.softmax(-1)
        labels, _ = model.tokenizer.decode(pred)
    text = labels[0].strip().replace(" ", "")
    sec = parse_clock(text)
    return sec, text


def second_frame_indices(second_sec, fps, frame_count):
    base = int(second_sec * fps)
    step = fps / OCR_CHECKS_PER_SEC
    max_idx = max(0, frame_count - 1)
    out = []
    for k in range(OCR_CHECKS_PER_SEC):
        idx = int(base + round(k * step))
        idx = max(0, min(idx, max_idx))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def ocr_vote_for_second(cap, model, transform, roi, second_sec, fps, frame_count):
    reads = []
    last_frame = None
    for idx in second_frame_indices(second_sec, fps, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        last_frame = frame
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        sec, text = parseq_predict_clock(model, transform, crop)
        actual_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        actual_video_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        reads.append(
            {
                "frame": frame,
                "crop": crop,
                "sec": sec,
                "text": text,
                "frame_idx": max(0, actual_frame_idx),
                "video_time_sec": max(0.0, actual_video_sec),
            }
        )

    valid = [r for r in reads if r["sec"] is not None]
    if not valid:
        return None, "", None, 0, 0, last_frame, None, None

    counts = Counter(r["sec"] for r in valid)
    voted_sec, hit_count = counts.most_common(1)[0]
    if hit_count < MIN_SAME_READS_FOR_MATCH:
        return None, "", None, hit_count, len(valid), last_frame, None, None

    picked = next(r for r in valid if r["sec"] == voted_sec)
    return (
        voted_sec,
        picked["text"],
        picked["crop"],
        hit_count,
        len(valid),
        last_frame,
        picked["frame_idx"],
        picked["video_time_sec"],
    )


def main():
    model, transform = load_parseq_tiny()

    all_events = load_events_from_excel(EXCEL_PATH, SHEET_NAME)
    if not all_events:
        raise RuntimeError("No events found in Excel sheet for annotation.")

    annotations = load_json_list(ANNOTATIONS_PATH)
    matched_ids = {a.get("row_id") for a in annotations if isinstance(a, dict)}
    pending = [e for e in all_events if e["row_id"] not in matched_ids]
    pending_index = {}
    for e in pending:
        pending_index.setdefault((e["period"], e["clock_sec"]), []).append(e)

    state = load_state(STATE_PATH)
    if isinstance(state.get("windows"), list) and state["windows"]:
        windows = []
        for item in state["windows"]:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                windows.append((int(item[0]), int(item[1]), int(item[2])))
        if windows:
            print("Resuming with quarter windows from scan_state.json")
        else:
            windows = input_quarter_windows()
    else:
        windows = input_quarter_windows()

    plan = build_sampling_plan(windows)
    if not plan:
        raise RuntimeError("No seconds to process from quarter windows.")
    start_index = int(state.get("next_plan_index", 0))
    if start_index < 0 or start_index >= len(plan):
        start_index = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("FPS unreadable")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        raise RuntimeError("Frame count unreadable")

    roi = None
    if isinstance(state.get("roi"), list) and len(state["roi"]) == 4:
        roi = tuple(int(v) for v in state["roi"])
        print(f"Resuming with ROI={roi} from scan_state.json")
    if roi is None:
        first_period, first_sec = plan[start_index]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_sec * fps))
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read frame for initial ROI selection.")
        roi = pick_roi_on_frame(first_frame)
        if roi is None:
            raise RuntimeError("ROI selection canceled.")

    print(
        f"Loaded {len(all_events)} events ({len(pending)} pending). "
        f"Start index={start_index}/{len(plan)} ROI={roi}"
    )
    print("Hotkeys: r=repick ROI, q=stop safely")

    stop_requested = False
    for i in range(start_index, len(plan)):
        period, t_sec = plan[i]
        sec, ocr_text, crop, vote_hits, valid_reads, frame, voted_frame_idx, voted_video_sec = ocr_vote_for_second(
            cap, model, transform, roi, t_sec, fps, frame_count
        )
        if frame is None:
            break
        if voted_frame_idx is None:
            voted_frame_idx = int(t_sec * fps)
        if voted_video_sec is None:
            voted_video_sec = float(t_sec)

        matched_now = pending_index.get((period, sec), []) if sec is not None else []
        if matched_now:
            pending_index[(period, sec)] = []
            for event in matched_now:
                out = {
                    "row_id": event["row_id"],
                    "period": event["period"],
                    "clock": event["clock"],
                    "code": event["code"],
                    "suggested_label": event["suggested_label"],
                    "team": event["team"],
                    "player": event["player"],
                    "cap": event["cap"],
                    "video_time_sec": voted_video_sec,
                    "video_time_hhmmss": format_hhmmss(voted_video_sec),
                    "frame_idx": voted_frame_idx,
                    "sample_time_sec": t_sec,
                    "sample_time_hhmmss": format_hhmmss(t_sec),
                    "ocr_text": ocr_text,
                    "vote_hits": vote_hits,
                    "valid_reads": valid_reads,
                }
                annotations.append(out)
                print(
                    f"Matched row_id={event['row_id']} (P{event['period']} {event['clock']}) "
                    f"at {format_hhmmss(voted_video_sec)} frame={voted_frame_idx}"
                )
            save_json_atomic(ANNOTATIONS_PATH, annotations)

        if (i % PRINT_EVERY) == 0:
            remaining = sum(len(v) for v in pending_index.values())
            print(
                f"[{i}/{len(plan)}] t={format_hhmmss(t_sec)} P={period} OCR='{ocr_text}' "
                f"sec={sec} vote={vote_hits}/{valid_reads} pending={remaining}"
            )

        vis = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"P{period} t={format_hhmmss(t_sec)} OCR={ocr_text} vote={vote_hits}/{valid_reads}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Scan", cv2.resize(vis, (1280, int(vis.shape[0] * 1280 / vis.shape[1]))))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            print("Manual ROI re-pick requested.")
            new_roi = pick_roi_on_frame(frame)
            if new_roi is not None:
                roi = new_roi
                print(f"Updated ROI to {roi}")
        elif key == ord("q"):
            print("Stop requested. Saving state and exiting.")
            stop_requested = True

        save_state(
            STATE_PATH,
            {
                "windows": [list(w) for w in windows],
                "next_plan_index": i + 1,
                "roi": list(roi),
            },
        )
        if stop_requested:
            break

    cap.release()
    cv2.destroyAllWindows()
    remaining = sum(len(v) for v in pending_index.values())
    print(f"Done. Matched annotations: {len(annotations)}. Remaining pending events: {remaining}")
    print(f"Annotations written to: {ANNOTATIONS_PATH}")
    print(f"Scan state written to: {STATE_PATH}")


if __name__ == "__main__":
    main()
