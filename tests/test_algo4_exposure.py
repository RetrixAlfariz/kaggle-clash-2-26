import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_algo4_exposure import whole_family_eligible


class FamilyEligibilityTests(unittest.TestCase):
    def test_exposed_member_excludes_whole_family(self):
        families = {'pair': ['a', 'b'], 'single': ['c']}
        self.assertEqual(whole_family_eligible(families, {'a': True, 'b': False, 'c': True}), {'single': ['c']})

    def test_unknown_member_is_not_admitted(self):
        self.assertEqual(whole_family_eligible({'pair': ['a', 'b']}, {'a': True, 'b': None}), {})

    def test_missing_history_fails_closed(self):
        with self.assertRaises(KeyError):
            whole_family_eligible({'pair': ['a', 'b']}, {'a': True})

    def test_all_eligible_members_retained(self):
        families = {'pair': ['a', 'b']}
        self.assertEqual(whole_family_eligible(families, {'a': True, 'b': True}), families)


if __name__ == '__main__':
    unittest.main()
