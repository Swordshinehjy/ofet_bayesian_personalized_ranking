from preprocess_new import validate_wildcard_count


class TestValidateWildcardCount:
    def test_two_wildcards(self):
        ok, count = validate_wildcard_count("*c1ccccc1*")
        assert ok is True
        assert count == 2

    def test_no_wildcard(self):
        ok, count = validate_wildcard_count("c1ccccc1")
        assert ok is False
        assert count == 0

    def test_one_wildcard(self):
        ok, count = validate_wildcard_count("*c1ccccc1")
        assert ok is False
        assert count == 1

    def test_three_wildcards(self):
        ok, count = validate_wildcard_count("*C*C*C")
        assert ok is False
        assert count == 3

    def test_custom_expected_count(self):
        ok, count = validate_wildcard_count("*C*C*C", expected_count=3)
        assert ok is True
        assert count == 3

    def test_empty_string(self):
        ok, count = validate_wildcard_count("")
        assert ok is False
        assert count == 0

    def test_real_polymer_smiles(self):
        smiles = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
        ok, count = validate_wildcard_count(smiles)
        assert ok is True
        assert count == 2
