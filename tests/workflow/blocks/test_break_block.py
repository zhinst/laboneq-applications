from laboneq_applications.workflow import blocks


class TestBreakLoopBlock:
    def test_str(self):
        assert str(blocks.BreakLoopBlock()) == "break_()"


def test_break_no_context():
    assert blocks.break_() is None
