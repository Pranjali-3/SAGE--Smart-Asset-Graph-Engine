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
# NASA Sensor Metadata - Only 7 Important Sensors
# ==========================================================

SENSOR_META = {
    "sensor_2":  {"name": "Core Temperature",    "unit": "degR"},
    "sensor_5":  {"name": "Engine Temperature",  "unit": "degR"},
    "sensor_7":  {"name": "Rotor Speed",         "unit": "rpm"},
    "sensor_11": {"name": "Oil Temperature",     "unit": "degR"},
    "sensor_12": {"name": "Oil Pressure",        "unit": "psia"},
    "sensor_17": {"name": "Rotor Vibration",     "unit": "mil"},
    "sensor_21": {"name": "Engine Efficiency",   "unit": "ratio"},
}

# Only 7 important sensors
KEY_SENSORS = [
    "sensor_2", "sensor_5", "sensor_7",
    "sensor_11", "sensor_12", "sensor_17", "sensor_21"
]

# ==========================================================
# Helper: detect trend direction (natural language)
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

    if diff_pct > 20:
        return "increased significantly"
    elif diff_pct > 5:
        return "increased gradually"
    elif diff_pct < -20:
        return "decreased significantly"
    elif diff_pct < -5:
        return "decreased gradually"
    else:
        return "remained stable"


# ==========================================================
# Helper: describe sensor in natural language (no numbers)
# ==========================================================

def _describe_sensor_natural(sensor_name, values):
    """
    Describe sensor behavior in natural language without raw numbers.
    """

    trend = _detect_trend(values)
    mean_val = np.mean(values)
    std_val = np.std(values)

    if "Core Temperature" in sensor_name or "Engine Temperature" in sensor_name:
        if "increased significantly" in trend:
            return f"{sensor_name} was at a high level and {trend} over the monitoring period"
        elif "increased gradually" in trend:
            return f"{sensor_name} was at a normal level and {trend} over the monitoring period"
        elif "decreased significantly" in trend:
            return f"{sensor_name} dropped significantly over the monitoring period"
        elif "decreased gradually" in trend:
            return f"{sensor_name} {trend} over the monitoring period"
        else:
            return f"{sensor_name} remained at a stable operating level"

    elif "Rotor Speed" in sensor_name:
        if "decreased" in trend:
            return f"{sensor_name} dropped, indicating reduced rotational performance"
        elif "increased" in trend:
            return f"{sensor_name} rose, showing increased load"
        else:
            return f"{sensor_name} maintained steady rotational speed"

    elif "Vibration" in sensor_name:
        if "increased" in trend:
            return f"{sensor_name} increased, suggesting potential mechanical wear"
        elif "decreased" in trend:
            return f"{sensor_name} decreased, indicating improved stability"
        else:
            return f"{sensor_name} remained within normal limits"

    elif "Oil Temperature" in sensor_name or "Oil Pressure" in sensor_name:
        if "decreased" in trend:
            return f"{sensor_name} dropped, possibly due to lubrication degradation"
        elif "increased" in trend:
            return f"{sensor_name} increased, indicating thermal stress"
        else:
            return f"{sensor_name} remained within normal operating range"

    elif "Engine Efficiency" in sensor_name:
        if "decreased" in trend:
            return f"{sensor_name} declined, indicating performance degradation"
        elif "increased" in trend:
            return f"{sensor_name} improved over the monitoring period"
        else:
            return f"{sensor_name} remained consistent"

    else:
        if "stable" in trend:
            return f"{sensor_name} remained stable throughout the period"
        else:
            return f"{sensor_name} {trend} over the monitoring period"


# ==========================================================
# Helper: health status explanation
# ==========================================================

def _health_status_explanation(status, rul):
    """
    Explain health status in natural language with reason.
    """

    if status == "critical":
        return (
            f"The engine is in critical condition because remaining useful life is "
            f"below 40 cycles. Immediate maintenance is recommended to prevent failure."
        )
    elif status == "warning":
        return (
            f"The engine is showing warning signs because remaining useful life is "
            f"between 40 and 120 cycles. Close monitoring and scheduled maintenance "
            f"should be considered."
        )
    else:
        return (
            f"The engine is operating normally because remaining useful life is "
            f"above 120 cycles. Continue routine monitoring."
        )


# ==========================================================
# Helper: describe sensors block (natural language only)
# ==========================================================

def _describe_sensors_block(engine_df, sensor_cols):
    """
    Build natural-language descriptions for key sensors.
    No raw numbers - only trend descriptions.
    """

    descriptions = []

    for col in sensor_cols:

        if col not in SENSOR_META:
            continue

        meta = SENSOR_META[col]
        name = meta["name"]

        values = engine_df[col].values

        desc = _describe_sensor_natural(name, values)

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
# NASA TXT Extraction - One Document Per Engine
# ==========================================================

def extract_nasa_txt(path):
    """
    Parse NASA C-MAPSS dataset into ONE descriptive document per engine.
    Each document contains:
    - Engine metadata
    - Overall health status with explanation
    - Trend summaries for 7 key sensors
    - Natural language descriptions (no raw numbers)
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

    # Only keep 7 important sensors
    active_sensors = [s for s in KEY_SENSORS if s in dataframe.columns]

    text_blocks = []

    for engine_id in sorted(dataframe["engine_id"].unique()):

        engine_df = dataframe[dataframe["engine_id"] == engine_id].copy()
        total_cycles = len(engine_df)

        # Get health status
        if is_train:
            max_cycle = int(engine_df["cycle"].max())
            rul = max_cycle - int(engine_df["cycle"].max())
            if max_cycle == 0:
                status = "critical"
            else:
                pct_remaining = rul / max_cycle * 100
                if pct_remaining > 70:
                    status = "healthy"
                elif pct_remaining > 30:
                    status = "warning"
                else:
                    status = "critical"
            rul_text = f"The engine has approximately {rul} cycles of remaining useful life."
        else:
            status = "unknown"
            rul = None
            rul_text = "Remaining useful life is unknown because this is a test trajectory."

        # Health status explanation
        if status != "unknown":
            status_explanation = _health_status_explanation(status, rul)
        else:
            status_explanation = "Health status cannot be determined for test trajectories."

        # Sensor trend summaries - natural language only
        sensor_descriptions = _describe_sensors_block(engine_df, active_sensors)

        # Build overall trend summary
        trend_parts = []
        for col in active_sensors:
            if col not in SENSOR_META:
                continue
            meta = SENSOR_META[col]
            values = engine_df[col].values
            trend = _detect_trend(values)
            if "stable" not in trend:
                trend_parts.append(f"{meta['name']} {trend}")

        if trend_parts:
            trend_summary = (
                "Notable trends observed during the monitoring period: "
                + "; ".join(trend_parts) + "."
            )
        else:
            trend_summary = "All monitored sensors remained stable throughout the observation period."

        # Build the complete document for this engine
        document_parts = [
            f"Predictive Maintenance Analysis Report - {dataset_name}",
            f"Engine {engine_id} - Total {total_cycles} operating cycles observed.",
            "",
            "Health Assessment:",
            f"Status: {status.upper()}.",
            f"{rul_text}",
            f"{status_explanation}",
            "",
            "Sensor Monitoring Results:",
            " ".join(sensor_descriptions),
            "",
            "Trend Analysis:",
            trend_summary
        ]

        if is_train:
            document_parts.extend([
                "",
                "Conclusion:",
                f"Based on {total_cycles} operating cycles, the engine's health status is {status}.",
                f"{status_explanation}"
            ])

        description = "\n".join(document_parts)

        # Rich metadata
        chunk_dict = {
            "text": description,
            "source": filename,
            "engine": int(engine_id),
            "cycle_start": 1,
            "cycle_end": total_cycles,
            "status": status,
            "rul": rul,
            "dataset": dataset_name,
            "is_test": not is_train,
            "is_summary": True,
            "chunk_id": len(text_blocks)
        }

        text_blocks.append(chunk_dict)

    logging.info(
        f"Created {len(text_blocks)} engine documents "
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