#include "../src/unicode.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static bool parse_nfc_codepoint_column(const std::string & field, std::vector<uint32_t> & output) {
    std::istringstream stream(field);
    std::string token;
    while (stream >> token) {
        try {
            size_t parsed = 0;
            const unsigned long value = std::stoul(token, &parsed, 16);
            if (parsed != token.size() || value > 0x10FFFFul) {
                return false;
            }
            output.push_back(static_cast<uint32_t>(value));
        } catch (const std::exception &) {
            return false;
        }
    }
    return true;
}

static bool run_nfc_normalization_test(const char * path) {
    std::ifstream input(path);
    if (!input) {
        fprintf(stderr, "unable to open NormalizationTest.txt: %s\n", path);
        return false;
    }

    size_t rows = 0;
    std::string line;
    while (std::getline(input, line)) {
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#' || line[first] == '@') {
            continue;
        }
        const size_t comment = line.find('#', first);
        if (comment != std::string::npos) {
            line.resize(comment);
        }

        std::vector<std::string> fields;
        size_t begin = 0;
        while (begin <= line.size()) {
            const size_t end = line.find(';', begin);
            fields.emplace_back(line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1;
        }
        if (fields.size() < 5) {
            fprintf(stderr, "malformed NormalizationTest row %zu\n", rows + 1);
            return false;
        }

        std::vector<std::vector<uint32_t>> columns(5);
        for (size_t column = 0; column < 5; ++column) {
            if (!parse_nfc_codepoint_column(fields[column], columns[column])) {
                fprintf(stderr, "malformed NormalizationTest row %zu column c%zu\n", rows + 1, column + 1);
                return false;
            }
        }

        ++rows;
        const std::pair<size_t, size_t> checks[] = { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 3 }, { 4, 3 } };
        for (const auto & check : checks) {
            const auto actual = unicode_cpts_normalize_nfc(columns[check.first]);
            if (actual != columns[check.second]) {
                fprintf(stderr, "NFC NormalizationTest mismatch at row %zu column c%zu (expected c%zu)\n",
                        rows, check.first + 1, check.second + 1);
                return false;
            }
        }
    }

    if (input.bad() || rows == 0) {
        fprintf(stderr, "NormalizationTest input is empty or unreadable\n");
        return false;
    }
    fprintf(stdout, "Unicode NFC NormalizationTest: %zu rows checked\n", rows);
    return true;
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        fprintf(stderr, "usage: test-unicode [NormalizationTest.txt]\n");
        return 2;
    }

    {
        const std::vector<std::string> regex_exprs = {
            "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
        };
        const std::vector<std::string> expected = { " ~", "foo" };
        const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

        if (actual != expected) {
            fprintf(stderr, "unexpected split:");
            for (const auto & piece : actual) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            return 1;
        }
    }

    {
        const std::vector<std::string> regex_exprs = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::string input = "ab\u200ccd ef\u200dgh cafe\u0301";
        const std::vector<std::string> expected = {
            "ab\u200ccd",
            " ef\u200dgh",
            " cafe\u0301",
        };

        try {
            const auto actual = unicode_regex_split(input, regex_exprs, false);

            if (actual != expected) {
                fprintf(stderr, "unexpected K2-Horizon split:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }
                fprintf(stderr, "\n");
                return 1;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "K2-Horizon regex split threw exception: %s\n", e.what());
            return 1;
        }
    }


    {
        const std::vector<std::string> llama3_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> k2_horizon_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> inputs = {
            "Hello, world!",
            "can't won't we're they'll",
            "123 1234 1234567",
            "Hello!!!\\nNext line",
            "alpha beta gamma",
            " café résumé",
        };

        for (const auto & input : inputs) {
            const auto llama3 = unicode_regex_split(input, llama3_regex, false);
            const auto k2 = unicode_regex_split(input, k2_horizon_regex, false);

            if (llama3 != k2) {
                fprintf(stderr, "K2-Horizon diverged from Llama 3 for ordinary input: %s\n", input.c_str());

                fprintf(stderr, "Llama 3:");
                for (const auto & piece : llama3) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nK2-Horizon:");
                for (const auto & piece : k2) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return 1;
            }
        }
    }


    {
        const std::vector<std::string> k2_horizon_regex = {
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        auto check_k2 = [&] (
                const char * name,
                const std::string & input,
                const std::vector<std::string> & expected) {
            const auto actual = unicode_regex_split(input, k2_horizon_regex, false);

            if (actual != expected) {
                fprintf(stderr, "K2-Horizon reference mismatch for %s\n", name);

                fprintf(stderr, "expected:");
                for (const auto & piece : expected) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nactual:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return false;
            }

            return true;
        };

        if (!check_k2(
                "ZWNJ",
                "ab\u200ccd",
                { "ab\u200ccd" })) {
            return 1;
        }

        if (!check_k2(
                "ZWJ",
                "ef\u200dgh",
                { "ef\u200dgh" })) {
            return 1;
        }

        if (!check_k2(
                "combined ZWNJ/ZWJ",
                "ab\u200ccd ef\u200dgh",
                { "ab\u200ccd", " ef\u200dgh" })) {
            return 1;
        }

        // The reference tokenizer NFC-normalizes cafe + combining acute
        // to the composed form before pre-tokenization.
        if (!check_k2(
                "NFC accent",
                "caf\u00e9",
                { "caf\u00e9" })) {
            return 1;
        }

        if (!check_k2(
                "Persian ZWNJ",
                "\u0645\u06cc\u200c\u0631\u0648\u0645",
                { "\u0645\u06cc\u200c\u0631\u0648\u0645" })) {
            return 1;
        }

        if (!check_k2(
                "Devanagari ZWJ",
                "\u0915\u094d\u200d\u0937",
                { "\u0915\u094d\u200d\u0937" })) {
            return 1;
        }

        if (!check_k2(
                "contractions",
                "can't won't we're they'll",
                { "can", "'t", " won", "'t", " we", "'re", " they", "'ll" })) {
            return 1;
        }

        if (!check_k2(
                "Unicode contraction",
                "'\u017fa",
                { "'\u017f", "a" })) {
            return 1;
        }

        if (!check_k2(
                "empty input",
                "",
                { })) {
            return 1;
        }

        if (!check_k2(
                "numbers",
                "123 1234 1234567",
                { "123", " ", "123", "4", " ", "123", "456", "7" })) {
            return 1;
        }

        if (!check_k2(
                "punctuation/newline",
                "Hello!!!\nNext line",
                { "Hello", "!!!\n", "Next", " line" })) {
            return 1;
        }
    }

    // NFC is tested independently of the regex splitter so these cases also
    // cover marks that the K2 regex intentionally keeps inside one fragment.
    {
        auto check_nfc = [&] (
                const char * name,
                const std::string & input,
                const std::string & expected) {
            const auto actual = unicode_cpts_normalize_nfc(unicode_cpts_from_utf8(input));
            const auto expected_cpts = unicode_cpts_from_utf8(expected);
            if (actual != expected_cpts) {
                fprintf(stderr, "NFC mismatch for %s\n", name);
                return false;
            }
            return true;
        };

        if (!check_nfc("decomposed e acute", "cafe\u0301", "caf\u00e9")) {
            return 1;
        }
        if (!check_nfc("canonical combining-class reordering", "q\u0307\u0323", "q\u0323\u0307")) {
            return 1;
        }
        // U+0958 is a Full_Composition_Exclusion: its canonical pair must
        // remain decomposed in NFC.
        if (!check_nfc("composition exclusion", "\u0915\u093c", "\u0915\u093c")) {
            return 1;
        }
        if (!check_nfc("Hangul LVT", "\u1100\u1161\u11a8", "\uac01")) {
            return 1;
        }

        // HF tokenizers 0.22.1 uses Unicode 9 normalization data.
        // Unicode 13 composition remains decomposed because UCD 9 has no pair.
        if (!check_nfc("HF Unicode 9 Dives Akuru", "\U00011935\U00011930", "\U00011935\U00011930")) {
            return 1;
        }
        // U+1DFA is Unicode 14. UCD 9 treats it as unknown, so HF keeps input order.
        if (!check_nfc("HF Unicode 9 unknown combining mark", "\u059a\U00001dfa", "\u059a\U00001dfa")) {
            return 1;
        }
    }

    if (argc == 2 && !run_nfc_normalization_test(argv[1])) {
        return 1;
    }

    return 0;
}
