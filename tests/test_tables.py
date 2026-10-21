from common.tables import align_row_decimals


def test_align_row_decimals():
    row = ["ARPU (monthly)", "1,240.06", "1,240", "1,239.99"]
    assert align_row_decimals(row) == ["ARPU (monthly)", "1,240.06", "1,240.00", "1,239.99"]


def test_align_leaves_integer_rows_and_percents():
    assert align_row_decimals(["New Customers", "1,679", "1,835"]) == ["New Customers", "1,679", "1,835"]
    assert align_row_decimals(["GM %", "53.81%", "54%", "1.5"]) == ["GM %", "53.81%", "54%", "1.5"]
    assert align_row_decimals(["EBIT", "-1,200", "(3,000)", "250.50"]) == ["EBIT", "-1,200.00", "(3,000.00)", "250.50"]
