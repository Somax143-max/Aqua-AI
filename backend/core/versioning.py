"""
AquaProtect-AI: Component Versioning & Release Tracking Engine
Provides unified version metadata attached to every inference run:
- Model Architecture Version
- Preprocessor Version
- Hydroacoustic Physics Version
- Dataset Partition Version
"""

from typing import Dict, Any

class SystemVersioning:
    """Provides release versioning stamps for audits."""

    VERSIONS = {
        "system_release": "AquaProtect-AI v2.2-Defense-Edition",
        "neural_model": "v2.2-DeepResNet-scSE-Attention-7Class",
        "preprocessor": "v2.0-Lee-Frost-SonarCLAHE",
        "physics_engine": "v2.2-Mackenzie-Snell-AHSA",
        "dataset_partition": "v3.0-HeldOut-IndependentSplit",
        "api_schema": "v2.2.0-JSON"
    }

    @classmethod
    def get_version_manifest(cls) -> Dict[str, str]:
        return cls.VERSIONS.copy()

    @classmethod
    def attach_versions_to_results(cls, results_dict: Dict[str, Any]) -> Dict[str, Any]:
        results_dict["system_versioning"] = cls.get_version_manifest()
        return results_dict