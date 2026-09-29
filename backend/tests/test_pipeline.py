"""
Automated Test & Verification Suite for AquaProtect-AI
Tests:
- Sonar Physics (SRC, shadow height, nadir detection)
- Speckle Filtering & Preprocessing (Lee, Frost, CLAHE)
- Hydrographic Geotagging (WGS-84 projection, UTM)
- Deep Detection & Physics Verification (AHSA)
- Multi-Format Report Generation (PDF, JSON, CSV, GeoJSON)
"""

import os
import json
import unittest
import numpy as np
import cv2

from core.sonar_physics import SonarPhysics
from core.preprocessor import SonarPreprocessor
from core.geotagging import SonarGeotagger
from core.motion_compensation import MotionCompensator
from core.sonar_parser import SonarDataParser
from core.mission_planner import OceanCleanupMissionPlanner
from core.svp_raytracer import SoundVelocityProfiler
from core.material_classifier import AcousticMaterialClassifier
from core.debris_drift_tracker import DebrisDriftTracker
from models.physics_filter import AcousticPhysicsFilter
from models.detector import SonarDebrisDetector, DEBRIS_CLASSES
from models.ghost_net_analyzer import GhostNetAnalyzer
from models.edge_profiler import EdgeHardwareProfiler
from models.deep_ensemble import DeepSonarDeterminationModel
from models.ablation_benchmark import ScientificAblationBenchmark
from data.sonar_generator import SonarDataGenerator
from data.sample_missions import load_mission_data
from reports.report_generator import HazardReportGenerator
from reports.s57_enc_exporter import S57ENCExporter

class TestAquaProtectAI(unittest.TestCase):

    def setUp(self):
        self.physics = SonarPhysics(max_slant_range=75.0, samples_per_channel=500)
        self.preprocessor = SonarPreprocessor()
        self.geotagger = SonarGeotagger(max_slant_range_m=75.0)
        self.physics_filter = AcousticPhysicsFilter()
        self.detector = SonarDebrisDetector(confidence_threshold=0.45)
        self.reports = HazardReportGenerator(output_dir="tests_output")

    def test_01_sonar_physics_shadow_height(self):
        """Test relief height computation from acoustic shadow geometry"""
        altitude = 12.0 # meters
        shadow_len = 15.0 # meters
        slant_range = 35.0 # meters
        h = self.physics.calculate_shadow_height(altitude, shadow_len, slant_range)
        # Expected: h = (12 * 15) / (35 + 15) = 180 / 50 = 3.6 m
        self.assertAlmostEqual(h, 3.6, places=1)
        self.assertGreater(h, 0.0)
        self.assertLess(h, altitude)

    def test_02_water_column_detection(self):
        """Test detection of nadir blind zone bottom return"""
        # Create synthetic ping line with dark water column followed by seabed echo
        ping = np.ones(500, dtype=np.uint8) * 15
        ping[60:] = np.random.normal(120, 15, 440).clip(0, 255).astype(np.uint8)
        idx, alt_m = self.physics.detect_water_column(ping)
        self.assertGreater(idx, 40)
        self.assertLess(idx, 100)
        self.assertGreater(alt_m, 5.0)

    def test_03_slant_to_ground_correction(self):
        """Test 2D slant range to ground range rectification"""
        raw_strip = np.random.normal(110, 20, (50, 1000)).clip(0, 255).astype(np.uint8)
        corrected = self.physics.slant_to_ground_range(raw_strip, altitude_m=12.0)
        self.assertEqual(corrected.shape, raw_strip.shape)
        self.assertEqual(corrected.dtype, np.uint8)

    def test_04_preprocessor_filters(self):
        """Test Lee filter, Frost filter, and Sonar-CLAHE enhancement"""
        noisy = np.random.rayleigh(scale=35.0, size=(100, 200)).clip(0, 255).astype(np.uint8)
        
        # Test Lee filter
        lee = self.preprocessor.apply_lee_filter(noisy)
        self.assertEqual(lee.shape, noisy.shape)
        
        # Test Frost filter
        frost = self.preprocessor.apply_frost_filter(noisy)
        self.assertEqual(frost.shape, noisy.shape)
        
        # Test full denoise + CLAHE
        denoised, enhanced = self.preprocessor.denoise_and_enhance(noisy, filter_type="lee")
        self.assertEqual(enhanced.shape, noisy.shape)
        
        # Test colormap
        cmap_img = self.preprocessor.apply_sonar_colormap(enhanced, colormap_name="bronze")
        self.assertEqual(len(cmap_img.shape), 3)
        self.assertEqual(cmap_img.shape[2], 3)

    def test_05_hydrographic_geotagging(self):
        """Test conversion of sonar pixel coordinate to WGS-84 Lat/Lon"""
        nav = {
            "latitude": 13.0827,
            "longitude": 80.3705,
            "heading": 90.0, # Heading East
            "altitude": 12.0,
            "gps_accuracy_m": 1.0
        }
        # Detect on Starboard side (dx > 0)
        bbox = (750, 200, 40, 30)
        geo = self.geotagger.geotag_detection(bbox, waterfall_width_px=1000, nav_record=nav)
        
        self.assertEqual(geo["channel"], "STARBOARD")
        self.assertGreater(geo["ground_range_m"], 0.0)
        # Heading East (90 deg) -> Starboard bearing is South (180 deg)
        self.assertAlmostEqual(geo["bearing_deg"], 180.0, places=1)
        # Bearing South means target latitude should be slightly south of vessel lat
        self.assertLess(geo["target_lat"], nav["latitude"])
        self.assertAlmostEqual(geo["target_lon"], nav["longitude"], places=3)

    def test_06_synthetic_mission_and_detection(self):
        """Test complete end-to-end mission generation, detection, and physics validation"""
        mission = load_mission_data("mission_chennai_niot")
        sonar_img = mission["sonar_image"]
        nav = mission["nav_telemetry"]
        
        # Warmup pass
        _ = self.detector.detect(sonar_img, nav_record=nav, filter_type="lee")
        # Timed fast inference pass
        res = self.detector.detect(sonar_img, nav_record=nav, filter_type="lee")
        self.assertIn("detections", res)
        self.assertGreaterEqual(res["total_detected"], 1)
        self.assertEqual(res["inference_mode"], "fast")
        self.assertGreater(res["fps"], 2.0) # Real-time speed check (>2.0 FPS required)
        
        # Check first detection structure
        first_det = res["detections"][0]
        self.assertIn("class_name", first_det)
        self.assertIn("confidence", first_det)
        self.assertIn("dimensions", first_det)
        self.assertIn("geotag", first_det)
        # Batch 2 & 3 & P0: Verify neural classification and model checkpoint status
        self.assertEqual(res["model_status"], "TRAINED_WEIGHTS_LOADED")
        self.assertIn("v2.", res["model_version"])
        self.assertIn("NEURAL_MODEL", first_det["classification_source"])
        self.assertEqual(len(first_det["class_probabilities"]), 7)
        self.assertIn("bayesian_certainty", first_det)

        # Evaluation mode verification (Monte Carlo dropout multi-sampling)
        eval_res = self.detector.detect(
            sonar_img, nav_record=nav, filter_type="lee", compute_uncertainty=True, num_mc_samples=2
        )
        self.assertEqual(eval_res["inference_mode"], "evaluation")
        self.assertIn("bayesian_certainty", eval_res["detections"][0])

    def test_07_multi_format_reports(self):
        """Test PDF, JSON, CSV, and GeoJSON report generation"""
        mission = load_mission_data("mission_chennai_niot")
        res = self.detector.detect(mission["sonar_image"], nav_record=mission["nav_telemetry"])
        
        json_path = self.reports.generate_json_report(
            mission["nav_telemetry"], res["detections"], {"fps": res["fps"], "latency_ms": res["latency_ms"]}
        )
        self.assertTrue(os.path.exists(json_path))
        self.assertGreater(os.path.getsize(json_path), 100)
        
        csv_path = self.reports.generate_csv_report(mission["nav_telemetry"], res["detections"])
        self.assertTrue(os.path.exists(csv_path))
        self.assertGreater(os.path.getsize(csv_path), 50)
        
        geojson_path = self.reports.generate_geojson(mission["nav_telemetry"], res["detections"])
        self.assertTrue(os.path.exists(geojson_path))
        self.assertGreater(os.path.getsize(geojson_path), 100)
        
        pdf_path = self.reports.generate_pdf_report(
            mission["nav_telemetry"], res["detections"], {"fps": res["fps"], "latency_ms": res["latency_ms"]}
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_08_motion_compensation(self):
        """Test acoustic data dropout detection, inpainting, and heave destriping"""
        compensator = MotionCompensator()
        # Create strip with artificial black dropout line
        strip = np.random.normal(120, 20, (100, 600)).clip(0, 255).astype(np.uint8)
        strip[45:48, :] = 2 # Simulate severe AUV roll dropout
        
        # Test detection
        dropouts = compensator.detect_data_dropouts(strip)
        self.assertTrue(np.any(dropouts))
        
        # Test inpainting repair
        repaired, count = compensator.inpaint_data_dropouts(strip, dropouts)
        self.assertGreater(count, 0)
        # Check that the black row was restored to ambient level
        repaired_mean = np.mean(repaired[46, :])
        self.assertGreater(repaired_mean, 50.0)

        # Test heave destriping
        destriped = compensator.apply_heave_destriping(repaired)
        self.assertEqual(destriped.shape, strip.shape)

    def test_09_ghost_net_filament_and_eri(self):
        """Test ghost net skeletonization, Entanglement Risk Index (ERI), and ROV waypoints"""
        analyzer = GhostNetAnalyzer()
        # Synthetic net patch with filament lines
        patch = np.zeros((60, 60), dtype=np.uint8)
        cv2.line(patch, (10, 10), (50, 50), 220, 2)
        cv2.line(patch, (10, 50), (50, 10), 220, 2)
        cv2.circle(patch, (30, 30), 12, 240, -1)

        skel, length = analyzer.skeletonize_net_filament(patch)
        self.assertGreater(length, 20.0)
        self.assertEqual(skel.shape, patch.shape)

        eri_res = analyzer.calculate_entanglement_risk_index(
            {"length_m": 8.0, "width_m": 6.0, "relief_height_m": 2.2},
            length, water_depth_m=25.0
        )
        self.assertIn("eri_score", eri_res)
        self.assertGreaterEqual(eri_res["eri_score"], 0.0)
        self.assertLessEqual(eri_res["eri_score"], 100.0)

        waypoints = analyzer.generate_rov_cutting_waypoints(skel, 13.0827, 80.3705, 2.2)
        self.assertGreater(len(waypoints), 0)
        self.assertIn("cutter_lat", waypoints[0])

    def test_10_cleanup_mission_planner(self):
        """Test autonomous ROV recovery route optimization using 2-opt TSP"""
        planner = OceanCleanupMissionPlanner()
        sample_dets = [
            {"id": "HAZ-001", "class_name": "Ghost Net", "severity": "CRITICAL", "geotag": {"target_lat": 13.085, "target_lon": 80.375}},
            {"id": "HAZ-002", "class_name": "Chemical Drum", "severity": "HIGH", "geotag": {"target_lat": 13.090, "target_lon": 80.371}},
            {"id": "HAZ-003", "class_name": "Shipwreck", "severity": "HIGH", "geotag": {"target_lat": 13.080, "target_lon": 80.380}}
        ]
        plan = planner.plan_recovery_mission(13.082, 80.370, sample_dets)
        self.assertEqual(plan["total_waypoints"], 4) # 1 vessel origin + 3 targets
        self.assertGreater(plan["total_distance_nm"], 0.0)
        self.assertGreater(plan["fuel_consumption_liters"], 0.0)

    def test_11_edge_profiler_and_onnx(self):
        """Test ONNX model export and hardware profile extraction"""
        profiler = EdgeHardwareProfiler(output_dir="tests_output")
        model = self.detector.model
        onnx_path = profiler.export_to_onnx(model, filename="test_model.onnx")
        self.assertTrue(os.path.exists(onnx_path))
        self.assertGreater(os.path.getsize(onnx_path), 25000)

        hw_profiles = profiler.get_hardware_profiles()
        self.assertEqual(len(hw_profiles), 4)

    def test_12_sound_velocity_profile_and_ray_tracing(self):
        """Test Mackenzie sound speed equation and Snell's law acoustic ray tracer"""
        # Benchmark ground truth: at 20 C, 35 ppt, 100 m depth
        c_val = SoundVelocityProfiler.mackenzie_sound_speed(20.0, 35.0, 100.0)
        self.assertGreater(c_val, 1515.0)
        self.assertLess(c_val, 1530.0)

        # Generate realistic Indian Ocean SVP
        svp = SoundVelocityProfiler.generate_indian_ocean_svp(max_depth_m=80.0)
        self.assertEqual(len(svp), 50)

        # Trace acoustic ray
        ray_res = SoundVelocityProfiler.trace_acoustic_ray(
            launch_angle_deg=35.0,
            transducer_depth_m=20.0,
            seafloor_depth_m=50.0,
            svp_profile=svp
        )
        self.assertIn("benthic_ground_range_m", ray_res)
        self.assertGreater(ray_res["benthic_ground_range_m"], 0.0)
        self.assertIn("refraction_offset_m", ray_res)

    def test_13_material_impedance_and_ts(self):
        """Test Acoustic Material Impedance and Target Strength classifier"""
        classifier = AcousticMaterialClassifier()
        dummy_crop = np.full((32, 32), 220, dtype=np.uint8)

        # Shipwreck should classify as steel
        res_steel = classifier.classify_target_material(dummy_crop, "Shipwreck / Hull Debris")
        self.assertEqual(res_steel["material_key"], "STEEL_FERROUS")
        self.assertGreater(res_steel["acoustic_impedance_mrayls"], 40.0)

        # Ghost net should classify as synthetic polymer
        res_net = classifier.classify_target_material(dummy_crop, "Ghost Fishing Net")
        self.assertEqual(res_net["material_key"], "SYNTHETIC_POLYMER")
        self.assertLess(res_net["acoustic_impedance_mrayls"], 5.0)

    def test_14_debris_drift_and_temporal_change(self):
        """Test Hydrodynamic Debris Drift and Multi-Temporal survey differencing"""
        # 48-hour hydrodynamic drift forecast
        drift_res = DebrisDriftTracker.predict_debris_drift(
            lat=13.0827, lon=80.3705, class_name="Ghost Fishing Net", current_speed_knots=1.5
        )
        self.assertIn("predicted_lat_48h", drift_res)
        self.assertGreater(drift_res["total_drift_displacement_m"], 10.0)

        # Temporal survey matching
        t0 = [{"id": "H-01", "class_name": "Shipwreck", "geotag": {"target_lat": 13.080, "target_lon": 80.370}}]
        t1 = [
            {"id": "H-01", "class_name": "Shipwreck", "geotag": {"target_lat": 13.08002, "target_lon": 80.37002}},
            {"id": "H-02", "class_name": "Chemical Drum", "geotag": {"target_lat": 13.085, "target_lon": 80.375}}
        ]
        comp = DebrisDriftTracker.compare_temporal_surveys(t0, t1)
        self.assertEqual(comp["summary"]["static_count"], 1)
        self.assertEqual(comp["summary"]["new_deposit_count"], 1)

    def test_15_s57_enc_nautical_chart_export(self):
        """Test IHO S-57 / S-100 Electronic Navigational Chart export"""
        exporter = S57ENCExporter(output_dir="tests_output")
        nav = {"latitude": 13.0827, "longitude": 80.3705, "depth": 42.0}
        dets = [
            {
                "id": "HAZ-001",
                "class_name": "Ghost Fishing Net",
                "confidence": 92.5,
                "geotag": {"target_lat": 13.084, "target_lon": 80.372},
                "dimensions": {"relief_height_m": 2.4}
            }
        ]
        res = exporter.export_s57_layer(nav, dets)
        self.assertTrue(os.path.exists(res["geojson_path"]))
        self.assertTrue(os.path.exists(res["spec_path"]))
        self.assertEqual(res["total_hazards_charted"], 1)

    def test_16_ablation_benchmark_and_scse_model(self):
        """Test Scientific Ablation Benchmark and DeepSonarDeterminationModel with scSE attention"""
        bench = ScientificAblationBenchmark.get_benchmark_results()
        self.assertEqual(len(bench["models"]), 4)

        import torch
        model = DeepSonarDeterminationModel(num_classes=6)
        x = torch.randn(2, 1, 64, 64)
        logits, masks = model(x)
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(masks.shape, (2, 1, 64, 64))

        # Fast deterministic prediction
        fast_out = model.predict_fast(x)
        self.assertEqual(len(fast_out["predictions"]), 2)
        self.assertIn("predictive_variance", fast_out["predictions"][0])

        # Bayesian uncertainty
        bayes = model.predict_with_bayesian_uncertainty(x, num_mc_samples=3)
        self.assertEqual(len(bayes["predictions"]), 2)
        self.assertIn("predictive_variance", bayes["predictions"][0])

    def test_17_motion_benchmark_and_geotag_validation(self):
        """Test motion robustness benchmarking and geodetic uncertainty ellipses (Batches 8 & 9)"""
        mission = load_mission_data("mission_chennai_niot")
        sonar_img = mission["sonar_image"]
        
        # Test motion robustness benchmark
        mc = MotionCompensator()
        mc_bench = mc.benchmark_motion_robustness(sonar_img, num_dropouts=5)
        self.assertIn("psnr_gain_db", mc_bench)
        self.assertIn("dropout_recovery_rate_pct", mc_bench)
        self.assertGreaterEqual(mc_bench["dropout_recovery_rate_pct"], 80.0)

        # Test geodetic uncertainty ellipse
        ellipse = self.geotagger.estimate_uncertainty_ellipse(25.0, gps_accuracy_m=1.0)
        self.assertIn("semi_major_m", ellipse)
        self.assertIn("semi_minor_m", ellipse)
        self.assertGreater(ellipse["semi_major_m"], 0.0)

        # Test missing navigation telemetry handling
        no_nav_res = self.geotagger.geotag_detection((500, 200, 30, 20), 1000, None)
        self.assertFalse(no_nav_res["georeferenced"])
        self.assertIsNone(no_nav_res["target_lat"])

    def test_18_input_validator_and_independent_evaluator(self):
        """Test robust defensive validator and independent metrics evaluation (Batches 13 & 16)"""
        from core.validator import SonarInputValidator
        
        # Test image validator
        # Generate realistic acoustic noise so it passes OOD Domain Validation
        valid_img = np.random.normal(120, 25, (100, 100)).astype(np.uint8)
        is_ok, msg, gray = SonarInputValidator.validate_sonar_image(valid_img)
        self.assertTrue(is_ok)
        self.assertEqual(gray.shape, (100, 100))

        # Test invalid image rejection
        bad_img = np.zeros((10, 10), dtype=np.uint8)
        is_ok, msg, _ = SonarInputValidator.validate_sonar_image(bad_img)
        self.assertFalse(is_ok)

        # Test nav telemetry validation & normalization
        bad_nav = {"latitude": "invalid", "depth": -5.0}
        is_nav_ok, nav_msg, cleaned_nav = SonarInputValidator.validate_nav_telemetry(bad_nav)
        self.assertFalse(is_nav_ok)
        self.assertGreater(cleaned_nav["depth"], 0.0)

        # Test independent evaluation report existence
        eval_report_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_reports", "independent_evaluation_report.json")
        self.assertTrue(os.path.exists(eval_report_path))
        with open(eval_report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        self.assertIn("overall_accuracy_pct", report_data)
        self.assertIn("per_class_metrics", report_data)
        self.assertIn("confusion_matrix", report_data)

if __name__ == "__main__":
    unittest.main()
