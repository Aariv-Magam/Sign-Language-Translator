import unittest

from translate_sign import format_translation_result, resolve_word_from_letters, resolve_window_mode, update_translation_text


class TranslationTests(unittest.TestCase):
    def test_resolve_window_mode_disables_window_without_display(self) -> None:
        self.assertFalse(resolve_window_mode(show_window=True, env={"DISPLAY": ""}))

    def test_resolve_window_mode_keeps_window_when_display_available(self) -> None:
        self.assertTrue(resolve_window_mode(show_window=True, env={"DISPLAY": ":0"}))

    def test_resolve_window_mode_honors_explicit_no_window(self) -> None:
        self.assertFalse(resolve_window_mode(show_window=False, env={"DISPLAY": ":0"}))

    def test_update_translation_text_adds_new_label_after_debounce(self) -> None:
        text, letter_buffer, last_label, last_change_time = update_translation_text([], [], ["A"], None, 0.0, 0.1)
        self.assertEqual(text, [])
        self.assertEqual(letter_buffer, ["A"])
        self.assertEqual(last_label, "A")

        text, letter_buffer, last_label, last_change_time = update_translation_text(text, letter_buffer, ["B"], "A", 0.1, 0.6, debounce_seconds=0.5)
        self.assertEqual(text, ["A"])
        self.assertEqual(letter_buffer, ["B"])
        self.assertEqual(last_label, "B")

    def test_format_translation_result_reports_final_text(self) -> None:
        self.assertEqual(format_translation_result(["A", "B"]), "Final translation: A B")
        self.assertEqual(format_translation_result([]), "No signs were detected.")

    def test_resolve_word_from_letters_matches_known_words(self) -> None:
        self.assertEqual(resolve_word_from_letters(["H", "E", "L", "L", "O"]), "HELLO")
        self.assertEqual(resolve_word_from_letters(["T", "H", "A", "N", "K", "Y", "O", "U"]), "THANK YOU")
        self.assertIsNone(resolve_word_from_letters(["A", "B", "C"]))


if __name__ == "__main__":
    unittest.main()
