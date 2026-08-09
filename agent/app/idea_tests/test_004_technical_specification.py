"""
Test 004: Technical Specification
Difficulty: 4/10 (Moderate)
Category: Technical Research
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "004",
        "test_name": "Technical Specification",
        "difficulty_level": "4/10",
        "category": "Technical Research",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find the USB-C Power Delivery 3.1 specification details: maximum wattage, maximum voltage, "
        "and revision number. Also find one product that supports this specification and confirm "
        "with the manufacturer's documentation. Provide all specifications with citation URLs."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Maximum wattage (number)",
        "Maximum voltage (number)",
        "Revision number",
        "Product name with manufacturer confirmation URL",
        "Citation URLs for all information",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All three specification values present (wattage, voltage, revision)",
        "Product identified with manufacturer URL",
        "At least 3 authoritative URLs cited",
        "Technical accuracy of values",
    ]


def validate_specs(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate specification values present."""
    final_text = extract_final_text(result).lower()
    has_wattage = bool(re.search(r"\b(240|250)\s*w(att)?s?\b", final_text))
    has_voltage = bool(re.search(r"\b(48|50)\s*v(olt)?s?\b", final_text))
    has_revision = bool(re.search(r"\b(3\.1|3\.1\.0|pd\s*3\.1)\b", final_text))
    checks = sum([has_wattage, has_voltage, has_revision])
    return {
        "check": "specifications",
        "passed": checks >= 2,
        "score": checks / 3.0,
        "has_wattage": has_wattage,
        "has_voltage": has_voltage,
        "has_revision": has_revision,
        "reason": f"Specs: wattage={has_wattage}, voltage={has_voltage}, revision={has_revision}",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate URLs present."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    passed = len(urls) >= 3
    return {
        "check": "url_count",
        "passed": passed,
        "score": min(1.0, len(urls) / 3.0),
        "url_count": len(urls),
        "reason": f"Found {len(urls)} URLs",
    }


def validate_product(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate product with manufacturer URL."""
    final_text = extract_final_text(result).lower()
    has_product = bool(re.search(r"\b(product|charger|adapter|device)\b", final_text))
    urls = re.findall(r"https?://[^\s)\\\"]+", extract_final_text(result))
    manufacturer_urls = [u for u in urls if any(x in u.lower() for x in ["manufacturer", "company", "official", ".com/", "product"])]
    has_manufacturer_url = len(manufacturer_urls) > 0
    return {
        "check": "product_manufacturer",
        "passed": has_product and has_manufacturer_url,
        "score": 0.5 if has_product else 0.0 + (0.5 if has_manufacturer_url else 0.0),
        "has_product": has_product,
        "has_manufacturer_url": has_manufacturer_url,
        "reason": f"Product: {has_product}, Manufacturer URL: {has_manufacturer_url}",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_specs, validate_urls, validate_product]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
