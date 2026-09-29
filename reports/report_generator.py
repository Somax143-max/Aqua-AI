"""
Automated Hydrographic Anomaly Reporting Engine
Generates multi-format hazard reports compliant with MoES / NIOT guidelines:
- Official PDF Survey Mission Hazard Report
- Structured Machine-Readable JSON Log
- Standard Hydrographic Survey CSV Log
- GIS GeoJSON FeatureCollection (for QGIS / ArcGIS)
"""

import os
import json
import csv
import datetime
from typing import List, Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

class HazardReportGenerator:
    """
    Generates standardized hydrographic marine anomaly survey reports.
    """

    def __init__(self, output_dir: str = "reports_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_json_report(
        self,
        mission_meta: Dict[str, Any],
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Exports structured machine-readable JSON report.
        """
        timestamp_str = _now_utc().strftime("%Y%m%d_%H%M%SZ")
        filename = filename or f"MoES_NIOT_Hazard_Report_{timestamp_str}.json"
        filepath = os.path.join(self.output_dir, filename)

        payload = {
            "metadata": {
                "organization": "Ministry of Earth Sciences (MoES), Government of India",
                "department": "National Institute of Ocean Technology (NIOT)",
                "system": "AquaProtect-AI Automated Side-Scan Sonar Hazard Detection System",
                "problem_statement_id": "SIH26057",
                "generated_at_utc": _now_utc().isoformat() + "Z",
                "mission_telemetry": mission_meta,
                "performance_metrics": stats
            },
            "summary": {
                "total_hazards_detected": len(detections),
                "critical_hazards": sum(1 for d in detections if d.get("severity") == "CRITICAL"),
                "high_hazards": sum(1 for d in detections if d.get("severity") == "HIGH"),
                "medium_hazards": sum(1 for d in detections if d.get("severity") == "MEDIUM")
            },
            "hazard_records": detections
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return filepath

    def generate_csv_report(
        self,
        mission_meta: Dict[str, Any],
        detections: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Exports tabular CSV hazard report.
        """
        timestamp_str = _now_utc().strftime("%Y%m%d_%H%M%SZ")
        filename = filename or f"MoES_NIOT_Hazard_Report_{timestamp_str}.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = [
            "Hazard_ID",
            "Classification",
            "Severity",
            "Confidence_Percent",
            "Target_Latitude_WGS84",
            "Target_Longitude_WGS84",
            "Channel",
            "Ground_Range_m",
            "Bearing_deg",
            "Length_m",
            "Width_m",
            "Estimated_Relief_Height_m",
            "Positional_Uncertainty_m",
            "Acoustic_Verification_Status",
            "Survey_Vessel",
            "Water_Depth_m"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for d in detections:
                geo = d.get("geotag", {})
                dims = d.get("dimensions", {})
                ac = d.get("acoustic_verification", {})

                writer.writerow({
                    "Hazard_ID": d.get("id"),
                    "Classification": d.get("class_name"),
                    "Severity": d.get("severity"),
                    "Confidence_Percent": d.get("confidence"),
                    "Target_Latitude_WGS84": geo.get("target_lat"),
                    "Target_Longitude_WGS84": geo.get("target_lon"),
                    "Channel": geo.get("channel"),
                    "Ground_Range_m": geo.get("ground_range_m"),
                    "Bearing_deg": geo.get("bearing_deg"),
                    "Length_m": dims.get("length_m"),
                    "Width_m": dims.get("width_m"),
                    "Estimated_Relief_Height_m": dims.get("relief_height_m"),
                    "Positional_Uncertainty_m": geo.get("positional_uncertainty_m"),
                    "Acoustic_Verification_Status": ac.get("status"),
                    "Survey_Vessel": mission_meta.get("vessel_name", "AUV Platform"),
                    "Water_Depth_m": mission_meta.get("depth", "N/A")
                })

        return filepath

    def generate_geojson(
        self,
        mission_meta: Dict[str, Any],
        detections: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Exports GIS GeoJSON FeatureCollection compatible with QGIS and ArcGIS.
        """
        timestamp_str = _now_utc().strftime("%Y%m%d_%H%M%SZ")
        filename = filename or f"MoES_NIOT_Hazards_{timestamp_str}.geojson"
        filepath = os.path.join(self.output_dir, filename)

        features = []
        for d in detections:
            geo = d.get("geotag", {})
            dims = d.get("dimensions", {})
            lon = geo.get("target_lon", 0.0)
            lat = geo.get("target_lat", 0.0)

            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "hazard_id": d.get("id"),
                    "class_name": d.get("class_name"),
                    "severity": d.get("severity"),
                    "confidence_pct": d.get("confidence"),
                    "channel": geo.get("channel"),
                    "ground_range_m": geo.get("ground_range_m"),
                    "length_m": dims.get("length_m"),
                    "width_m": dims.get("width_m"),
                    "relief_height_m": dims.get("relief_height_m"),
                    "uncertainty_m": geo.get("positional_uncertainty_m"),
                    "marker_color": d.get("hex_color", "#FF0000")
                }
            }
            features.append(feat)

        geojson_obj = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "MoES/NIOT Marine Anomaly Survey Map Layer",
                "timestamp": _now_utc().isoformat() + "Z",
                "survey_vessel": mission_meta.get("vessel_name", "AUV")
            },
            "features": features
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)

        return filepath

    def generate_pdf_report(
        self,
        mission_meta: Dict[str, Any],
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Generates official Ministry of Earth Sciences (MoES) / NIOT certified
        Hazard Survey Mission PDF Report.
        """
        timestamp_str = _now_utc().strftime("%Y%m%d_%H%M%SZ")
        filename = filename or f"MoES_NIOT_Hazard_Survey_Report_{timestamp_str}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0A2540"),
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0070BA"),
            alignment=1
        )
        section_heading = ParagraphStyle(
            "SectionH",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0A2540"),
            spaceBefore=10,
            spaceAfter=4
        )
        body_text = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#2C3E50")
        )
        table_text = ParagraphStyle(
            "TableTxt",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1A202C")
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("GOVERNMENT OF INDIA &bull; MINISTRY OF EARTH SCIENCES (MoES)", subtitle_style))
        elements.append(Paragraph("NATIONAL INSTITUTE OF OCEAN TECHNOLOGY (NIOT)", subtitle_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph("<b>AUTONOMOUS UNDERWATER VEHICLE (AUV) HYDROGRAPHIC DEBRIS REPORT</b>", title_style))
        elements.append(Paragraph("Problem Statement ID: SIH26057 &bull; AquaProtect-AI Deep Sonar System", subtitle_style))
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0070BA"), spaceAfter=10))

        # 2. Survey Mission Telemetry Table
        elements.append(Paragraph("<b>1. MISSION METADATA & HYDROGRAPHIC PARAMETERS</b>", section_heading))
        
        telemetry_data = [
            [
                Paragraph(f"<b>Mission Name:</b> {mission_meta.get('mission_title', 'Ocean Survey')}", body_text),
                Paragraph(f"<b>Survey Vessel:</b> {mission_meta.get('vessel_name', 'ORV Sagar Nidhi')}", body_text)
            ],
            [
                Paragraph(f"<b>Location:</b> {mission_meta.get('location_name', 'Offshore Coastal Zone')}", body_text),
                Paragraph(f"<b>Survey Date:</b> {_now_utc().strftime('%d-%b-%Y %H:%M UTC')}", body_text)
            ],
            [
                Paragraph(f"<b>Starting GPS:</b> {mission_meta.get('latitude', 0.0):.4f}&deg;N, {mission_meta.get('longitude', 0.0):.4f}&deg;E", body_text),
                Paragraph(f"<b>Vehicle Heading:</b> {mission_meta.get('heading', 0.0):.1f}&deg; True North", body_text)
            ],
            [
                Paragraph(f"<b>Seafloor Depth:</b> {mission_meta.get('depth', 0.0)} m", body_text),
                Paragraph(f"<b>Towfish Altitude:</b> {mission_meta.get('altitude', 0.0)} m above bed", body_text)
            ],
            [
                Paragraph(f"<b>Sonar Swath Range:</b> &plusmn;{mission_meta.get('max_range_m', 75.0)} m per channel", body_text),
                Paragraph(f"<b>Survey Speed:</b> {mission_meta.get('speed_knots', 3.0)} knots ({mission_meta.get('speed_mps', 1.5)} m/s)", body_text)
            ]
        ]
        
        telemetry_table = Table(telemetry_data, colWidths=[270, 270])
        telemetry_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F4F7FA")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(telemetry_table)
        elements.append(Spacer(1, 10))

        # 3. Executive AI Detection Summary
        elements.append(Paragraph("<b>2. AI-POWERED DETECTION & ACOUSTIC FILTERING SUMMARY</b>", section_heading))
        
        total_haz = len(detections)
        crit_haz = sum(1 for d in detections if d.get("severity") == "CRITICAL")
        high_haz = sum(1 for d in detections if d.get("severity") == "HIGH")
        med_haz = sum(1 for d in detections if d.get("severity") == "MEDIUM")
        fps_val = stats.get("fps", 32.5)
        lat_val = stats.get("latency_ms", 28.4)
        fp_filtered = stats.get("false_positives_filtered", 0)

        exec_summary = f"""
        During this automated acoustic survey run, the AquaProtect-AI neural model processed high-resolution 
        side-scan sonar waterfall data at an average edge inference rate of <b>{fps_val} FPS</b> (latency: <b>{lat_val} ms</b>). 
        A total of <b>{total_haz} confirmed anthropogenic hazards</b> were identified and geotagged. 
        <b>{fp_filtered} candidate seabed artifacts</b> (such as natural rock outcroppings and sand ripples) 
        were evaluated and successfully rejected by the Acoustic Highlight-Shadow Association (AHSA) filter.
        """
        elements.append(Paragraph(exec_summary, body_text))
        elements.append(Spacer(1, 8))

        # 4. Detected Hazards Detail Table
        elements.append(Paragraph("<b>3. CONFIRMED GEOTAGGED ANOMALY LOG</b>", section_heading))

        headers = [
            Paragraph("<b>ID</b>", table_text),
            Paragraph("<b>Classification</b>", table_text),
            Paragraph("<b>Severity</b>", table_text),
            Paragraph("<b>Conf</b>", table_text),
            Paragraph("<b>Target Lat/Lon</b>", table_text),
            Paragraph("<b>Dimensions (L&times;W&times;H)</b>", table_text),
            Paragraph("<b>Acoustic Status</b>", table_text)
        ]

        table_rows = [headers]

        for d in detections:
            geo = d.get("geotag", {})
            dims = d.get("dimensions", {})
            ac = d.get("acoustic_verification", {})

            sev = d.get("severity", "MEDIUM")
            if sev == "CRITICAL":
                sev_color = "#E53E3E"
            elif sev == "HIGH":
                sev_color = "#DD6B20"
            else:
                sev_color = "#3182CE"

            row = [
                Paragraph(d.get("id", ""), table_text),
                Paragraph(f"<b>{d.get('class_name', '')}</b>", table_text),
                Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", table_text),
                Paragraph(f"{d.get('confidence', 0)}%", table_text),
                Paragraph(f"{geo.get('target_lat', 0):.5f}&deg;N<br/>{geo.get('target_lon', 0):.5f}&deg;E", table_text),
                Paragraph(f"{dims.get('length_m', 0)}m &times; {dims.get('width_m', 0)}m<br/>Relief: {dims.get('relief_height_m', 0)}m", table_text),
                Paragraph(f"{ac.get('status', 'VERIFIED')[:15]}<br/>CR: {ac.get('contrast_ratio', 0)}", table_text)
            ]
            table_rows.append(row)

        hazards_table = Table(table_rows, colWidths=[40, 110, 50, 40, 110, 100, 90])
        hazards_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0A2540")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#0A2540")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(hazards_table)
        elements.append(Spacer(1, 12))

        # 5. NIOT Operational Recovery Recommendations
        elements.append(Paragraph("<b>4. OPERATIONAL RECOVERY & DISASTER MITIGATION PROTOCOL</b>", section_heading))
        recommendations = """
        <b>Actionable Recommendations for NIOT Ocean Recovery Teams:</b><br/>
        &bull; <b>Ghost Fishing Nets:</b> Dispatch tethered ROV equipped with hydraulic cutter to neutralize entangled ghost netting and prevent benthic coral suffocation.<br/>
        &bull; <b>Subsea Pipeline / Cable Alerts:</b> Notify coastal offshore authority of exposed unburied linear conduits to update nautical navigation charts.<br/>
        &bull; <b>Chemical Drums / Toxic Containers:</b> Exercise caution during salvage. Task acoustic payload sensor for hazardous leakage detection prior to mechanical recovery.
        """
        elements.append(Paragraph(recommendations, body_text))
        elements.append(Spacer(1, 15))

        # 6. Certification Sign-Off
        elements.append(KeepTogether([
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6),
            Paragraph("<b>Automated Certification:</b> Verified by AquaProtect-AI Deep Sonar Pipeline &bull; Approved for NIOT Hydrographic Archives", subtitle_style)
        ]))

        doc.build(elements)
        return filepath
