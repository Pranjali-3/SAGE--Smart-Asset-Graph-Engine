import fitz
import io
import os
import logging
import pandas as pd
import pytesseract

import numpy as np

from PIL import Image
from docx import Document

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Configure Tesseract
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================================================
# NASA Sensor Column Names
# ==========================================================

NASA_COLUMNS = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3"
] + [f"sensor_{i}" for i in range(1, 22)]

# ==========================================================
# NASA Sensor Metadata
# ==========================================================

SENSOR_META = {
    "sensor_1":  {"name": "Fan Speed",                "unit": "rpm"},
    "sensor_2":  {"name": "Core Temperature",         "unit": "degR"},
    "sensor_3":  {"name": "Compressor Pressure",      "unit": "psia"},
    "sensor_4":  {"name": "Fuel Flow",                "unit": "pps"},
    "sensor_5":  {"name": "Engine Temperature",       "unit": "degR"},
    "sensor_6":  {"name": "Air Pressure",             "unit": "psia"},
    "sensor_7":  {"name": "Rotor Speed",              "unit": "rpm"},
    "sensor_8":  {"name": "Exhaust Temperature",      "unit": "degR"},
    "sensor_9":  {"name": "Vibration",                "unit": "mil"},
    "sensor_10": {"name": "Cooling Pressure",         "unit": "psia"},
    "sensor_11": {"name": "Oil Temperature",          "unit": "degR"},
    "sensor_12": {"name": "Oil Pressure",             "unit": "psia"},
    "sensor_13": {"name": "Fuel Pressure",            "unit": "psia"},
    "sensor_14": {"name": "Compressor Temperature",   "unit": "degR"},
    "sensor_15": {"name": "Turbine Temperature",      "unit": "degR"},
    "sensor_16": {"name": "Bearing Temperature",      "unit": "degR"},
    "sensor_17": {"name": "Rotor Vibration",          "unit": "mil"},
    "sensor_18": {"name": "Exhaust Pressure",         "unit": "psia"},
    "sensor_19": {"name": "Fuel Valve Position",      "unit": "ratio"},
    "sensor_20": {"name": "Air Intake Flow",          "unit": "pps"},
    "sensor_21": {"name": "Engine Efficiency",        "unit": "ratio"},
}

# Sensors to always include (informative and variable)
KEY_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_9", "sensor_11", "sensor_12", "sensor_14",
    "sensor_15", "sensor_17", "sensor_21"
]

# ==========================================================
# Helper: detect trend direction
# ==========================================================

def _detect_trend(values):
    """
    Return a human-readable trend description.
    """

    if len(values) < 2:
        return "stable"

    first_half = np.mean(values[:len(values) // 2])
    second_half = np.mean(values[len(values) // 2:])

    diff_pct = (second_half - first_half) / (abs(first_half) + 1e-9) * 100

    if diff_pct > 5:
        return "increased"
    elif diff_pct < -5:
        return "decreased"
    else:
        return "remained stable"


# ==========================================================
# Helper: percentage-based health status
# ==========================================================

def _health_status(rul, max_rul):
    """
    Return status based on percentage of life consumed.
    """

    if max_rul == 0:
        return "critical"

    pct_remaining = rul / max_rul * 100

    if pct_remaining > 70:
        return "healthy"
    elif pct_remaining > 30:
        return "warning"
    else:
        return "critical"


# ==========================================================
# Helper: describe sensor block
# ==========================================================

def _describe_sensors(engine_df, sensor_cols, sensor_meta):
    """
    Build natural-language descriptions for key sensors.
    Includes trend detection across the window.
    """

    descriptions = []

    for col in sensor_cols:

        meta = sensor_meta.get(col, {"name": col, "unit": ""})

        name = meta["name"]

        unit = meta["unit"]

        values = engine_df[col].values

        mean_val = np.mean(values)

        trend = _detect_trend(values)

        if trend == "increased":
            desc = (
                f"{name} averaged {mean_val:.1f} {unit} "
                f"and {trend} over the window."
            )
        elif trend == "decreased":
            desc = (
                f"{name} averaged {mean_val:.1f} {unit} "
                f"and {trend} over the window."
            )
        else:
            desc = (
                f"{name} averaged {mean_val:.1f} {unit} "
                f"and {trend} over the window."
            )

        descriptions.append(desc)

    return descriptions

# ==========================================================
# PDF Extraction
# ==========================================================

def extract_pdf(path):
    """
    Extract text from PDF.
    Uses OCR for scanned pages.
    """

    document = fitz.open(path)

    text = ""

    for page in document:

        page_text = page.get_text()

        if page_text.strip():

            text += page_text

        else:

            pix = page.get_pixmap(dpi=300)

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            text += pytesseract.image_to_string(image)

    return text


# ==========================================================
# DOCX Extraction
# ==========================================================

def extract_docx(path):
    """
    Extract text from Word document.
    """

    document = Document(path)

    text = []

    for paragraph in document.paragraphs:

        text.append(paragraph.text)

    return "\n".join(text)


# ==========================================================
# Excel Extraction
# ==========================================================

def extract_excel(path):
    """
    Extract all sheets from Excel.
    """

    excel = pd.ExcelFile(path)

    text = ""

    for sheet in excel.sheet_names:

        dataframe = pd.read_excel(
            path,
            sheet_name=sheet
        )

        text += dataframe.to_string()

        text += "\n\n"

    return text


# ==========================================================
# CSV Extraction
# ==========================================================

def extract_csv(path):
    """
    Extract CSV file.
    """

    dataframe = pd.read_csv(path)

    logging.info(f"Rows    : {len(dataframe)}")
    logging.info(f"Columns : {len(dataframe.columns)}")

    return dataframe.to_string(index=False)


# ==========================================================
# Image OCR
# ==========================================================

def extract_image(path):
    """
    Extract text from image using OCR.
    """

    image = Image.open(path)

    return pytesseract.image_to_string(image)


# ==========================================================
# NASA TXT Extraction
# ==========================================================

def extract_nasa_txt(path):
    """
    Parse NASA C-MAPSS dataset into descriptive text blocks.
    Returns a list of dictionaries, each representing one
    engine window summary chunk.
    """

    filename = os.path.basename(path)
    is_train = filename.startswith("train")
    dataset_name = filename.replace(".txt", "")

    logging.info(f"Reading NASA Dataset: {filename}")

    dataframe = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NASA_COLUMNS
    )

    logging.info(f"Rows    : {len(dataframe)}")
    logging.info(f"Columns : {len(dataframe.columns)}")

    # Step 7: Identify sensors with variance (skip constant)
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    variance = dataframe[sensor_cols].var()
    variable_sensors = variance[variance > 1e-6].index.tolist()

    # Keep only KEY_SENSORS that are also variable
    active_sensors = [s for s in KEY_SENSORS if s in variable_sensors]

    # Step 1: Compute max_cycle for RUL
    max_cycles = dataframe.groupby("engine_id")["cycle"].max()
    global_max_rul = max_cycles.max()

    text_blocks = []

    window_size = 10

    for engine_id in sorted(dataframe["engine_id"].unique()):

        engine_df = dataframe[dataframe["engine_id"] == engine_id]
        engine_max_cycle = max_cycles[engine_id]
        total_cycles = len(engine_df)

        # Process every 10 cycles
        cycle_starts = list(range(1, total_cycles + 1, window_size))

        window_summaries = []

        for start_idx, cycle_start in enumerate(cycle_starts):

            cycle_end = min(cycle_start + window_size - 1, total_cycles)

            window_df = engine_df[
                (engine_df["cycle"] >= cycle_start) &
                (engine_df["cycle"] <= cycle_end)
            ]

            if window_df.empty:
                continue

            # Step 1: RUL handling
            if is_train:
                last_cycle_in_window = int(window_df["cycle"].max())
                rul = engine_max_cycle - last_cycle_in_window
                max_rul = engine_max_cycle
                status = _health_status(rul, max_rul)
                rul_text = (
                    f"Estimated Remaining Useful Life: {rul} cycles. "
                    f"Current health state: {status.capitalize()}."
                )
                status_text = f"Status: {status}."
            else:
                rul = None
                status = "unknown"
                rul_text = (
                    "Remaining Useful Life: unknown because "
                    "this is a test trajectory."
                )
                status_text = "Status: unknown (test trajectory)."

            # Step 3: Natural language sensor descriptions
            sensor_descs = _describe_sensors(
                window_df, active_sensors, SENSOR_META
            )

            # Step 4: Trend summary
            trend_parts = []
            for col in active_sensors:
                meta = SENSOR_META.get(col, {"name": col})
                values = window_df[col].values
                trend = _detect_trend(values)
                if trend != "remained stable":
                    trend_parts.append(
                        f"{meta['name']} {trend}"
                    )

            if trend_parts:
                trend_text = (
                    "Notable trends: "
                    + ", ".join(trend_parts) + "."
                )
            else:
                trend_text = "All key sensors remained stable."

            # Step 9: Dataset source
            header = (
                f"Dataset: {dataset_name}. "
                f"Engine {engine_id}. "
                f"Cycles {cycle_start} to {cycle_end}."
            )

            # Step 10: RUL explicit
            body_parts = [
                header,
                status_text,
                rul_text,
                " ".join(sensor_descs),
                trend_text
            ]

            description = " ".join(body_parts)

            # Step 6: Return dictionary with metadata
            chunk_dict = {
                "text": description,
                "source": filename,
                "engine": int(engine_id),
                "cycle_start": int(cycle_start),
                "cycle_end": int(cycle_end),
                "status": status,
                "dataset": dataset_name,
                "is_test": not is_train
            }

            window_summaries.append(chunk_dict)

        # Step 5: Engine summary
        last_window = engine_df.tail(window_size)

        if is_train:
            final_rul = engine_max_cycle - int(engine_df["cycle"].max())
            final_status = _health_status(final_rul, engine_max_cycle)
        else:
            final_rul = None
            final_status = "unknown"

        trend_parts = []
        for col in active_sensors:
            meta = SENSOR_META.get(col, {"name": col})
            values = engine_df[col].values
            trend = _detect_trend(values)
            if trend != "remained stable":
                trend_parts.append(
                    f"{meta['name']} gradually {trend}"
                )

        if trend_parts:
            trend_summary = "; ".join(trend_parts) + "."
        else:
            trend_summary = "All key sensors remained stable."

        if is_train:
            summary_rul_text = (
                f"Final health status: {final_status.capitalize()}. "
                f"Remaining life: {final_rul} cycles."
            )
        else:
            summary_rul_text = (
                "Final health status: unknown (test trajectory)."
            )

        summary = (
            f"Dataset: {dataset_name}. "
            f"Engine {engine_id} Summary. "
            f"Total cycles observed: {total_cycles}. "
            f"{summary_rul_text} "
            f"Sensor trends across all cycles: {trend_summary}"
        )

        summary_dict = {
            "text": summary,
            "source": filename,
            "engine": int(engine_id),
            "cycle_start": 1,
            "cycle_end": int(engine_max_cycle),
            "status": final_status,
            "dataset": dataset_name,
            "is_test": not is_train,
            "is_summary": True
        }

        text_blocks.extend(window_summaries)
        text_blocks.append(summary_dict)

    logging.info(
        f"Created {len(text_blocks)} blocks "
        f"from NASA dataset ({dataset_name})."
    )

    return text_blocks


# ==========================================================
# Generic TXT Extraction
# ==========================================================

def extract_txt(path):

    # NASA dataset - returns list of dicts
    if "FD00" in os.path.basename(path):
        return extract_nasa_txt(path)

    with open(path, "r", encoding="latin-1") as file:
        return file.read()


# ==========================================================
# Universal Document Loader
# ==========================================================

def ingest_document(path):
    """
    Load any supported document.
    """

    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":

        return extract_pdf(path)

    elif extension == ".docx":

        return extract_docx(path)

    elif extension == ".xlsx":

        return extract_excel(path)

    elif extension == ".csv":

        return extract_csv(path)

    elif extension in [".png", ".jpg", ".jpeg"]:

        return extract_image(path)

    elif extension == ".txt":

        return extract_txt(path)

    else:

        raise ValueError(
            f"Unsupported file type : {extension}"
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logging.info("Industrial Document Ingestion Started")

    folder = r"D:\hackathons\data\nasa\archive"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if os.path.isfile(path):

            print("\n" + "=" * 70)

            print(f"Reading : {file}")

            print("=" * 70)

            result = ingest_document(path)

            if isinstance(result, list):

                print(f"Blocks created: {len(result)}")

                if result:

                    print("\nFirst block preview:")

                    print(result[0]["text"][:500])

            else:

                print(result[:500])

    logging.info("Document Ingestion Completed")