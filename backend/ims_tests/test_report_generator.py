"""
Tests for server.report_generator -- the shared HTML/PDF report engine.

Includes a specific regression guard: an earlier version of this module
had a line-based edit accidentally delete the
`section_time_domain_plots(...)` call from build_iberian_report's
sections list, silently producing a report with NO embedded plots (and
about 1/20th the expected size) while still completing without error.
Every report builder's test below explicitly asserts the presence of
at least one embedded plot image, specifically to catch this class of
silent, no-exception content-loss bug again.
"""
import numpy as np

from ims_platform.server import iberian_scenario as isc
from ims_platform.server import gfm_current_limit_case as gcc
from ims_platform.server import multi_converter_fault_case as mcc
from ims_platform.server import report_generator as rg


def _assert_has_real_content(html_str: str, min_len: int = 20000):
    assert "data:image/png;base64" in html_str, "report is missing at least one embedded plot"
    assert len(html_str) > min_len, f"report suspiciously short ({len(html_str)} chars) -- likely a dropped section"


def test_iberian_report_has_embedded_plots_and_real_values():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r)
    _assert_has_real_content(html_str)
    assert "NON-RECOVERABLE" in html_str
    assert f"{r['summary']['max_v_es_sw']:.4f}" in html_str


def test_iberian_report_with_all_optional_sections():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    cf = isc.run_baseline_vs_dynamic_q_counterfactual()
    bm = isc.run_boundary_map({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    rec = isc.find_recommended_interventions({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r, counterfactual=cf, boundary=bm, recommendations=rec)
    _assert_has_real_content(html_str)
    # Narrowed to the Counterfactual section's own content specifically (not the whole
    # 7..9 span, which now legitimately includes the "C -- True IMS Geometry" section --
    # itself correctly showing "Not evaluated" here since ims_geometry=None was passed).
    counterfactual_section = html_str.split("7. Counterfactual")[1].split("8. Boundary")[0]
    assert "Not evaluated in this analysis." not in counterfactual_section


def test_iberian_report_without_optional_sections_says_not_evaluated():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r)
    assert "Not evaluated in this analysis." in html_str


def test_iberian_report_with_ims_geometry_section():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    ims_geo = isc.run_ims_geometry_analysis({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, roa_n_v=4, roa_n_timer=3)
    html_str = rg.build_iberian_report(r, ims_geometry=ims_geo)
    assert "True IMS Geometry" in html_str
    assert "Region of Attraction (ROA)" in html_str
    assert "basin" not in html_str.lower(), "must use Region of Attraction (ROA) terminology, not 'basin'"
    assert "gamma_ims" not in html_str  # internal key name should never leak into rendered HTML
    assert f"{ims_geo['gamma_ims']['gamma_ims']:.4f}" in html_str


def test_gfm_report_has_embedded_plots_and_real_values():
    r = gcc.run_stress_test({"Imax": 0.4})
    html_str = rg.build_gfm_report(r)
    _assert_has_real_content(html_str, min_len=15000)
    assert "NON-RECOVERABLE" in html_str
    assert f"{r['summary']['min_v_poc']:.4f}" in html_str


def test_gfm_report_recoverable_reference():
    r = gcc.run_stress_test({"Imax": 2.0})
    html_str = rg.build_gfm_report(r)
    assert "RECOVERABLE" in html_str
    _assert_has_real_content(html_str, min_len=15000)


def test_multiconverter_report_has_embedded_plots_and_real_values():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    html_str = rg.build_multiconverter_report(r)
    _assert_has_real_content(html_str, min_len=15000)
    assert "NON-RECOVERABLE" in html_str
    assert f"{r['summary']['min_v_busA']:.4f}" in html_str
    assert f"{r['summary']['min_v_busB']:.4f}" in html_str


def test_multiconverter_report_with_asymmetry_section():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    asym = mcc.verify_local_vs_remote_support_asymmetry()
    html_str = rg.build_multiconverter_report(r, asymmetry=asym)
    assert "Local vs. remote support" in html_str
    assert "GFM (anchor) support only" in html_str


def test_gfm_report_with_ims_geometry_shows_honest_not_applicable():
    r = gcc.run_stress_test({"Imax": 0.4})
    ims_geo = gcc.run_ims_geometry_analysis({"Imax": 0.4})
    html_str = rg.build_gfm_report(r, ims_geometry=ims_geo)
    assert "True IMS Geometry" in html_str
    assert "NOT APPLICABLE" in html_str
    assert "v_trip=999" in html_str
    assert "basin" not in html_str.lower()


def test_multiconverter_report_with_ims_geometry_shows_honest_not_applicable():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    ims_geo = mcc.run_ims_geometry_analysis(mcc.BASELINE_REFERENCE)
    html_str = rg.build_multiconverter_report(r, ims_geometry=ims_geo)
    assert "True IMS Geometry" in html_str
    assert "NOT APPLICABLE" in html_str
    assert "conv_gfm" in html_str and "conv_gfl" in html_str


def test_pdf_generation_produces_nonempty_valid_pdf():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r)
    pdf_bytes = rg.html_to_pdf_bytes(html_str)
    assert len(pdf_bytes) > 10000
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_generation_for_all_three_cases():
    r1 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    r2 = gcc.run_stress_test({"Imax": 0.4})
    r3 = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    for html_str in [rg.build_iberian_report(r1), rg.build_gfm_report(r2), rg.build_multiconverter_report(r3)]:
        pdf_bytes = rg.html_to_pdf_bytes(html_str)
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 10000


def test_report_never_shows_not_achievable_as_impossible():
    rec = isc.find_recommended_interventions({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r, recommendations=rec)
    assert "impossible" not in html_str.lower()
    assert "not achievable within tested range" in html_str


def test_pdf_uses_patched_wkhtmltopdf_and_has_footer_pagination():
    """
    Regression guard for a real, found bug: the system/apt-packaged
    wkhtmltopdf is an "unpatched Qt" build that SILENTLY ignores
    --footer-*/--header-* options (no exception, just a stderr warning),
    so page numbers were missing from every PDF. Fixed by bundling a
    patched-Qt wkhtmltopdf binary and pointing pdfkit at it explicitly.
    This test confirms the bundled binary is actually being found and
    used, not just present on disk.
    """
    config = rg._pdfkit_configuration()
    assert config is not False, "bundled patched wkhtmltopdf binary was not found -- PDF footers/page numbers will be silently dropped"


def test_pdf_footer_page_numbers_actually_render():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r)
    pdf_bytes = rg.html_to_pdf_bytes(html_str)
    import pdfplumber
    import io as _io
    with pdfplumber.open(_io.BytesIO(pdf_bytes)) as pdf:
        assert len(pdf.pages) > 1, "expected a multi-page report to actually test pagination"
        page1_text = pdf.pages[0].extract_text() or ""
        page2_text = pdf.pages[1].extract_text() or ""
    assert "IMS Platform Explorer" in page1_text
    assert "Page 1 of" in page1_text
    assert "Page 2 of" in page2_text
    # confirms the SAME total page count is reported consistently (not a fixed placeholder)
    total1 = page1_text.split("Page 1 of")[-1].strip().split()[0]
    total2 = page2_text.split("Page 2 of")[-1].strip().split()[0]
    assert total1 == total2 == str(len(pdfplumber.open(_io.BytesIO(pdf_bytes)).pages))


def test_failed_baseline_report_does_not_crash():
    """A case with no stable baseline must still produce a valid (if minimal) report, not raise."""
    bad_result = {"baseline_ok": False, "message": "test failure case", "disclaimer": "test disclaimer"}
    html_str = rg.build_iberian_report(bad_result)
    assert "test failure case" in html_str
    assert "<html>" in html_str


def test_report_sections_are_sequentially_numbered_iberian():
    """
    Structural regression guard for a real bug: section_* functions used to hard-code their own
    numbers (8, 11, 9, 10...), which broke sequential order once a case's included-section set
    varied. Numbering must now come entirely from _auto_number_sections, applied once to the
    fully-assembled document -- verified here by extracting every <h2> number in document order
    and checking it's exactly 1..N with no gaps or duplicates.
    """
    import re
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    cf = isc.run_baseline_vs_dynamic_q_counterfactual()
    bm = isc.run_boundary_map({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    rec = isc.find_recommended_interventions({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    ims_geo = isc.run_ims_geometry_analysis({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, roa_n_v=3, roa_n_timer=3)
    html_str = rg.build_iberian_report(r, counterfactual=cf, boundary=bm, recommendations=rec, ims_geometry=ims_geo)
    nums = [int(n) for n in re.findall(r"<h2>(\d+)\.", html_str)]
    assert nums == list(range(1, len(nums) + 1)), f"section numbers not sequential: {nums}"


def test_report_sections_are_sequentially_numbered_gfm():
    import re
    r = gcc.run_stress_test({"Imax": 0.4})
    sweep = gcc.run_imax_sweep()
    rec = gcc.find_recommended_interventions({"Imax": 0.4})
    ims_geo = gcc.run_ims_geometry_analysis({"Imax": 0.4})
    html_str = rg.build_gfm_report(r, sweep=sweep, ims_geometry=ims_geo, recommendations=rec)
    nums = [int(n) for n in re.findall(r"<h2>(\d+)\.", html_str)]
    assert nums == list(range(1, len(nums) + 1)), f"section numbers not sequential: {nums}"


def test_report_sections_are_sequentially_numbered_multiconverter():
    import re
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    asym = mcc.verify_local_vs_remote_support_asymmetry()
    rec = mcc.find_recommended_interventions(mcc.BASELINE_REFERENCE)
    ims_geo = mcc.run_ims_geometry_analysis(mcc.BASELINE_REFERENCE)
    html_str = rg.build_multiconverter_report(r, asymmetry=asym, ims_geometry=ims_geo, recommendations=rec)
    nums = [int(n) for n in re.findall(r"<h2>(\d+)\.", html_str)]
    assert nums == list(range(1, len(nums) + 1)), f"section numbers not sequential: {nums}"


def test_report_sections_sequential_even_with_minimal_optional_sections():
    """Numbering must stay sequential even when most optional sections are omitted (fewer <h2>s total)."""
    import re
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    html_str = rg.build_iberian_report(r)  # no optional extras at all
    nums = [int(n) for n in re.findall(r"<h2>(\d+)\.", html_str)]
    assert nums == list(range(1, len(nums) + 1)), f"section numbers not sequential: {nums}"


def test_iberian_report_generation_with_ims_geometry_shows_real_content_not_placeholder():
    """The flagship claim: with IMS geometry supplied, the Iberian report must show real numbers
    for the CORE geometric content (gamma_IMS, ROA, manifold definition) -- not a placeholder.
    (The pre-incident-warning sub-block legitimately says "Not evaluated" here since ims_warning
    wasn't supplied to this specific call -- that's a real, correct negative, not the bug being tested.)"""
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    ims_geo = isc.run_ims_geometry_analysis({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, roa_n_v=3, roa_n_timer=3)
    html_str = rg.build_iberian_report(r, ims_geometry=ims_geo)
    ims_section = html_str.split("True IMS Geometry</h2>")[1].split("<h2>")[0]
    assert "Region of Attraction (ROA)" in ims_section
    assert f"{ims_geo['gamma_ims']['gamma_ims']:.4f}" in ims_section
    assert "critical manifold" in ims_section.lower() or "manifold_definition" not in ims_section
