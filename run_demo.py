"""
AquaProtect-AI Master Launcher & Demonstration Suite
Problem Statement: SIH26057 - Ministry of Earth Sciences (MoES) / NIOT

Usage:
  python run_demo.py            # Launches the Interactive Streamlit Marine UI Dashboard
  python run_demo.py --cli      # Runs full pipeline in CLI mode across Indian Maritime Missions
  python run_demo.py --test     # Runs complete automated unit test suite
  python run_demo.py --bench    # Runs edge FPS and latency benchmark for AUV hardware
"""

import sys
import os
import argparse
import time
import subprocess
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_cli_demo():
    print("=" * 80)
    print("[*] AquaProtect-AI: Deep Sonar Marine Debris Detection System")
    print("    Ministry of Earth Sciences (MoES) & National Institute of Ocean Technology (NIOT)")
    print("    Problem Statement: SIH26057 | Disaster Management & Marine Conservation")
    print("=" * 80)

    from data.sample_missions import SAMPLE_MISSIONS, load_mission_data
    from models.detector import SonarDebrisDetector
    from reports.report_generator import HazardReportGenerator

    detector = SonarDebrisDetector(confidence_threshold=0.45)
    report_gen = HazardReportGenerator(output_dir="reports_output")

    for mission_key, meta in SAMPLE_MISSIONS.items():
        print(f"\n[+] Processing Mission: {meta['title']}")
        print(f"    Location: {meta['location_name']}")
        print(f"    Vessel: {meta['vessel_name']} | Depth: {meta['depth_m']}m | Altitude: {meta['altitude_m']}m")

        mission = load_mission_data(mission_key)
        sonar_img = mission["sonar_image"]
        nav = mission["nav_telemetry"]

        start_t = time.time()
        res = detector.detect(sonar_img, nav_record=nav, filter_type="lee")
        elapsed = (time.time() - start_t) * 1000.0

        detections = res["detections"]
        print(f"    >> Completed in {res['latency_ms']} ms ({res['fps']} FPS) | Total Detected: {len(detections)}")
        print(f"    >> Physics Filter Rejections (False Alarms): {res['false_positives_filtered']}")

        print("\n    Confirmed Hazards Detected:")
        print(f"    {'ID':<9} {'Classification':<24} {'Severity':<10} {'Conf':<6} {'Lat/Lon':<24} {'Dimensions (LxWxH)':<20} {'Status':<15}")
        print("    " + "-" * 112)
        for d in detections:
            geo = d["geotag"]
            dims = d["dimensions"]
            ac = d["acoustic_verification"]
            coord_str = f"{geo['target_lat']:.5f}N, {geo['target_lon']:.5f}E"
            dim_str = f"{dims['length_m']}x{dims['width_m']}x{dims['relief_height_m']}m"
            print(f"    {d['id']:<9} {d['class_name']:<24} {d['severity']:<10} {d['confidence']}%  {coord_str:<24} {dim_str:<20} {ac['status'][:15]:<15}")

        # Generate Reports
        stats = {"fps": res["fps"], "latency_ms": res["latency_ms"], "false_positives_filtered": res["false_positives_filtered"]}
        pdf_f = report_gen.generate_pdf_report(nav, detections, stats, filename=f"Report_{mission_key}.pdf")
        json_f = report_gen.generate_json_report(nav, detections, stats, filename=f"Report_{mission_key}.json")
        csv_f = report_gen.generate_csv_report(nav, detections, filename=f"Report_{mission_key}.csv")
        geo_f = report_gen.generate_geojson(nav, detections, filename=f"Report_{mission_key}.geojson")
        print(f"    >> Generated Official PDF Report: {pdf_f}")
        print(f"    >> Generated GIS GeoJSON Layer: {geo_f}")

    print("\n" + "=" * 80)
    print("✅ All Maritime Missions processed successfully!")
    print("   Output reports saved in: ./reports_output/")
    print("=" * 80)

def run_tests():
    print("[*] Running automated unit test suite...")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

def run_bench():
    print("[*] Running Edge Hardware Inference & Latency Benchmark...")
    from data.sample_missions import load_mission_data
    from models.detector import SonarDebrisDetector

    detector = SonarDebrisDetector(confidence_threshold=0.45)
    mission = load_mission_data("mission_chennai_niot")
    img = mission["sonar_image"]
    nav = mission["nav_telemetry"]

    # Warmup
    for _ in range(3):
        _ = detector.detect(img, nav_record=nav)

    # Benchmark 20 iterations
    latencies = []
    for i in range(20):
        t0 = time.time()
        _ = detector.detect(img, nav_record=nav)
        latencies.append((time.time() - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / avg_lat
    print(f"\n[+] Benchmark Results over 20 Acoustic Swaths:")
    print(f"    Average Latency: {avg_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    print(f"    Effective Throughput: {fps:.1f} FPS")
    print(f"    Edge Readiness: {'EXCELLENT (AUV Real-Time Compatible)' if fps > 10 else 'GOOD'}")

def run_plan_demo():
    print("[*] Running Autonomous Ocean Cleanup & ROV Recovery Mission Planner...")
    from data.sample_missions import load_mission_data
    from models.detector import SonarDebrisDetector
    from core.mission_planner import OceanCleanupMissionPlanner

    detector = SonarDebrisDetector(confidence_threshold=0.45)
    planner = OceanCleanupMissionPlanner()

    mission = load_mission_data("mission_chennai_niot")
    res = detector.detect(mission["sonar_image"], nav_record=mission["nav_telemetry"])
    plan = planner.plan_recovery_mission(
        mission["nav_telemetry"]["latitude"],
        mission["nav_telemetry"]["longitude"],
        res["detections"]
    )

    print(f"\n[+] Optimized Cleanup Trajectory (2-Opt TSP):")
    print(f"    Total Route Distance: {plan['total_distance_nm']} Nautical Miles ({plan['total_distance_km']} km)")
    print(f"    Total Mission Time:   {plan['total_mission_hours']} hours (Transit: {plan['transit_duration_hours']}h)")
    print(f"    Fuel Consumption:     {plan['fuel_consumption_liters']} Liters Marine Diesel")
    print(f"    Operational Cost:     INR {plan['estimated_operational_cost_inr']:,.2f}")
    print(f"    Route Savings:        {plan['savings_vs_unoptimized_pct']}% reduction vs unoptimized search")

    print("\n    Sequential Waypoint Itinerary:")
    for step in plan["itinerary"]:
        act = step["action"]
        dist = step["leg_distance_m"]
        print(f"    Step {step['step']:<2}: {step['latitude']:.5f}N, {step['longitude']:.5f}E (+{dist}m) -> {act}")

def run_export_demo():
    print("[*] Running Edge Model Export & Hardware Profiler...")
    from models.detector import SonarDebrisDetector
    from models.edge_profiler import EdgeHardwareProfiler

    detector = SonarDebrisDetector()
    profiler = EdgeHardwareProfiler(output_dir="models_output")

    path = profiler.export_to_onnx(detector.model)
    print(f"[+] Exported model successfully to: {path} ({os.path.getsize(path)/1024:.1f} KB)")

    print("\n[+] Target Edge Hardware Benchmarks:")
    print(f"    {'Platform':<36} {'Precision':<16} {'FPS':<8} {'Latency':<10} {'Power':<8} {'AUV Battery'}")
    print("    " + "-" * 90)
    for p in profiler.get_hardware_profiles():
        print(f"    {p['platform']:<36} {p['precision']:<16} {p['fps']:<8.1f} {p['latency_ms']:<7.1f}ms {p['power_watts']:<5.1f}W  {p['auv_battery_endurance_hours']} hrs")

def run_ablation_demo():
    print("[*] Running Scientific Ablation Study & Neural Benchmarking...")
    from models.ablation_benchmark import ScientificAblationBenchmark
    print("\n" + ScientificAblationBenchmark.generate_markdown_report())

def run_drift_demo():
    print("[*] Running Hydrodynamic Debris Drift & Temporal Change Simulation...")
    from core.debris_drift_tracker import DebrisDriftTracker
    res = DebrisDriftTracker.predict_debris_drift(
        lat=13.0827, lon=80.3705, class_name="Ghost Fishing Net", current_speed_knots=1.4, current_bearing_deg=55.0
    )
    print(f"\n[+] 48-Hour Hydrodynamic Drift Forecast for Ghost Net:")
    print(f"    Current Velocity:   {res['current_speed_knots']} knots @ {res['current_bearing_deg']}° Bearing")
    print(f"    Displacement Speed: {res['drift_velocity_mps']} m/s ({res['risk_category']})")
    print(f"    Total Shift (48h):  {res['total_drift_displacement_m']} meters")
    print(f"    Predicted 24h:      {res['predicted_lat_24h']:.5f}N, {res['predicted_lon_24h']:.5f}E")
    print(f"    Predicted 48h:      {res['predicted_lat_48h']:.5f}N, {res['predicted_lon_48h']:.5f}E")

def run_svp_demo():
    print("[*] Running Sound Velocity Profile & Snell's Law Ray-Tracing Simulation...")
    from core.svp_raytracer import SoundVelocityProfiler
    svp = SoundVelocityProfiler.generate_indian_ocean_svp(max_depth_m=80.0)
    ray = SoundVelocityProfiler.trace_acoustic_ray(
        launch_angle_deg=35.0, transducer_depth_m=18.0, seafloor_depth_m=50.0, svp_profile=svp
    )
    print(f"\n[+] Mackenzie Oceanographic Ray-Tracing Results:")
    print(f"    Surface Sound Speed:    {svp[0]['sound_speed_mps']} m/s (T={svp[0]['temp_c']}°C)")
    print(f"    Seabed Sound Speed:     {svp[-1]['sound_speed_mps']} m/s (T={svp[-1]['temp_c']}°C)")
    print(f"    Benthic Ground Range:   {ray['benthic_ground_range_m']} meters")
    print(f"    Two-Way Travel Time:    {ray['two_way_travel_time_ms']} ms")
    print(f"    Refraction Offset:      {ray['refraction_offset_m']} meters lateral deviation from straight-line")

def run_train_demo():
    print("[*] Training Ultra-High Accuracy Determination Model with scSE Attention...")
    from models.trainer import train_model
    train_model(epochs=10)

def launch_ui():
    print("[*] Launching AquaProtect-AI Marine UI Dashboard...")
    dashboard_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaProtect-AI Launcher")
    parser.add_argument("--cli", action="store_true", help="Run full CLI demonstration across missions")
    parser.add_argument("--test", action="store_true", help="Run automated unit test suite")
    parser.add_argument("--bench", action="store_true", help="Run edge inference benchmark")
    parser.add_argument("--plan-route", action="store_true", help="Run ROV cleanup mission route planner")
    parser.add_argument("--export-edge", action="store_true", help="Export model for Jetson edge deployment")
    parser.add_argument("--ablation", action="store_true", help="Run scientific ablation benchmark")
    parser.add_argument("--drift-sim", action="store_true", help="Run hydrodynamic debris drift simulation")
    parser.add_argument("--svp-demo", action="store_true", help="Run Sound Velocity Profile ray-tracing demo")
    parser.add_argument("--train", action="store_true", help="Train scSE attention model")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI Dashboard")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo()
    elif args.test:
        run_tests()
    elif args.bench:
        run_bench()
    elif args.plan_route:
        run_plan_demo()
    elif args.export_edge:
        run_export_demo()
    elif args.ablation:
        run_ablation_demo()
    elif args.drift_sim:
        run_drift_demo()
    elif args.svp_demo:
        run_svp_demo()
    elif args.train:
        run_train_demo()
    else:
        # Default: launch UI
        launch_ui()
