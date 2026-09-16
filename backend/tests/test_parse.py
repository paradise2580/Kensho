"""Parser tests.

Several of these are regressions for bugs found while building the corpus.
They are marked as such, because a test whose failure mode you have actually
seen is worth more than one written from imagination.
"""

from __future__ import annotations

import pytest

from kensho.corpus.parse import clean_inline, parse_markdown, split_front_matter
from kensho.tokenizer import HeuristicTokenCounter, script_of


class TestShortcodes:
    def test_glossary_tooltip_keeps_displayed_term(self):
        """REGRESSION: deleting shortcodes wholesale removed ~2,000 key terms.

        The tag carries the noun the user actually reads and searches for.
        Stripping it produced sentences like
        '各cloud-controller-managerは複数のを実装します' — grammatical debris
        with the subject removed.
        """
        src = (
            '複数の{{< glossary_tooltip text="コントローラー" '
            'term_id="controller" >}}を実装します。'
        )
        assert clean_inline(src) == "複数のコントローラーを実装します。"

    def test_glossary_tooltip_falls_back_to_term_id(self):
        src = 'A {{< glossary_tooltip term_id="StatefulSet" >}} manages Pods.'
        assert clean_inline(src) == "A StatefulSet manages Pods."

    def test_paired_shortcode_keeps_inner_prose(self):
        """Notes and cautions hold the caveats users ask about most."""
        src = "{{< note >}}\nThis caveat matters.\n{{< /note >}}"
        assert "This caveat matters." in clean_inline(src)

    def test_feature_state_becomes_readable_text(self):
        src = '{{< feature-state state="stable" for_k8s_version="v1.28" >}} Ready.'
        assert clean_inline(src).strip() == "(feature state: stable, v1.28) Ready."

    def test_param_is_dropped_not_named(self):
        """Emitting the variable name would inject 'Version version required'."""
        assert clean_inline('Version {{< param "version" >}} required.').split() == [
            "Version", "required.",
        ]

    def test_unknown_shortcode_leaves_no_residue(self):
        assert "{{" not in clean_inline('{{< code_sample file="x.yaml" >}}')


class TestFrontMatter:
    def test_extracted_and_removed(self):
        fm, body = split_front_matter("---\ntitle: Pods\nweight: 20\n---\n\nBody text.\n")
        assert fm["title"] == "Pods"
        assert body.strip() == "Body text."

    def test_absent_front_matter_is_not_an_error(self):
        fm, body = split_front_matter("Just text.\n")
        assert fm == {} and body.startswith("Just")

    def test_malformed_yaml_does_not_raise(self):
        fm, _ = split_front_matter("---\ntitle: [unclosed\n---\nBody\n")
        assert fm == {}


class TestStructure:
    def test_heading_hierarchy_becomes_section_path(self):
        page = parse_markdown(
            "---\ntitle: T\n---\n# Top\nintro\n## Middle\nmid text\n### Deep\ndeep text\n"
        )
        paths = {b.section_path: b.text for b in page.blocks}
        assert ("Top",) in paths
        assert ("Top", "Middle") in paths
        assert ("Top", "Middle", "Deep") in paths

    def test_sibling_heading_pops_the_stack(self):
        page = parse_markdown("# A\n## B1\nx\n## B2\ny\n")
        assert ("A", "B2") in {b.section_path for b in page.blocks}
        assert ("A", "B1", "B2") not in {b.section_path for b in page.blocks}

    def test_hash_inside_code_fence_is_not_a_heading(self):
        """REGRESSION: shell comments and YAML keys would fabricate sections."""
        page = parse_markdown("# Real\n```bash\n# not a heading\nkubectl get pods\n```\n")
        assert all(p.section_path == ("Real",) for p in page.blocks)
        assert "kubectl get pods" in page.body_text

    def test_link_text_survives_target_removal(self):
        assert clean_inline("see [the Pod docs](/docs/concepts/pods/)") == "see the Pod docs"


class TestTokenCounter:
    def test_japanese_and_english_use_different_ratios(self):
        """REGRESSION guard: one ratio for both scripts would quarter the
        semantic content of Japanese chunks and look like a retriever fault."""
        c = HeuristicTokenCounter()
        ja, en = "ポッドは複数のコンテナを持てます", "A Pod can hold several containers"
        assert len(ja) < len(en)          # fewer characters
        assert c.count(ja) > c.count(en)  # but more tokens

    def test_empty_is_zero(self):
        assert HeuristicTokenCounter().count("") == 0

    @pytest.mark.parametrize("text,expected", [
        ("ポッドとコンテナ", "cjk"),
        ("Pods and containers", "latin"),
    ])
    def test_script_detection(self, text, expected):
        assert script_of(text) == expected
