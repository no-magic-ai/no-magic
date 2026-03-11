"""
no-magic repository overview v2 — 65-second animated montage for LinkedIn.
Silent autoplay optimized: text overlays + animations, no voiceover.
"""

from manim import *
import numpy as np


# === COLOR PALETTE ===
BG_COLOR = "#0d1117"       # GitHub dark
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"
ACCENT_RED = "#f85149"
ACCENT_TEAL = "#39d353"
TEXT_DIM = "#8b949e"
TEXT_BRIGHT = "#e6edf3"


class NoMagicOverview(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        self.act1_title()
        self.act2_montage()
        self.act3_structure()
        self.act4_cta()

    # =================================================================
    # ACT 1: Title + Tagline (0–7s)
    # =================================================================
    def act1_title(self):
        # Repo name — large, bold
        title = Text("no-magic", font_size=72, weight=BOLD, color=TEXT_BRIGHT)
        # Python icon substitute — a simple ">" prompt glyph
        prompt = Text(">>>", font_size=48, color=ACCENT_GREEN)
        prompt.next_to(title, LEFT, buff=0.4)
        title_group = VGroup(prompt, title).move_to(UP * 0.5)

        # Version badge — subtle top-right of title
        version = Text("v2.0", font_size=18, color=ACCENT_PURPLE)
        version.next_to(title, UR, buff=0.1)

        tagline = Text(
            'model.fit() isn\'t an explanation',
            font_size=36, color=ACCENT_ORANGE, slant=ITALIC
        )
        tagline.next_to(title_group, DOWN, buff=0.5)

        subtitle = Text(
            "41 algorithms from scratch  ·  pure Python  ·  zero dependencies",
            font_size=22, color=TEXT_DIM
        )
        subtitle.next_to(tagline, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=DOWN * 0.3), FadeIn(prompt, shift=RIGHT * 0.3), run_time=1.0)
        self.play(FadeIn(version, scale=0.5), run_time=0.3)
        self.play(Write(tagline), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.wait(2.0)

        self.play(
            *[FadeOut(mob, shift=UP * 0.5) for mob in [title_group, version, tagline, subtitle]],
            run_time=0.6
        )
        self.wait(0.3)

    # =================================================================
    # ACT 2: Algorithm Montage (7–47s)
    # ~5 seconds per algorithm, 8 algorithms
    # =================================================================
    def act2_montage(self):
        self.montage_tokenizer()
        self.montage_vit()
        self.montage_moe()
        self.montage_speculative()
        self.montage_lstm()
        self.montage_diffusion()
        self.montage_mcts()
        self.montage_gpt()

    # --- microtokenizer: text → tokens ---
    def montage_tokenizer(self):
        label = self._montage_label("microtokenizer.py", "01-foundations")

        raw = Text("understanding", font_size=44, color=TEXT_BRIGHT)
        raw.move_to(UP * 0.5)
        self.play(FadeIn(label), Write(raw), run_time=0.8)
        self.wait(0.3)

        # Split into BPE-style subwords
        pieces = ["under", "##stand", "##ing"]
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]
        tokens = VGroup()
        for piece, color in zip(pieces, colors):
            tok = VGroup(
                RoundedRectangle(
                    width=len(piece) * 0.35 + 0.6, height=0.7,
                    corner_radius=0.15, color=color, fill_opacity=0.25,
                    stroke_width=2
                ),
                Text(piece, font_size=28, color=color)
            )
            tokens.add(tok)

        tokens.arrange(RIGHT, buff=0.25)
        tokens.move_to(DOWN * 0.8)

        # Arrow from raw to tokens
        arrow = Arrow(raw.get_bottom(), tokens.get_top(), buff=0.2, color=TEXT_DIM, stroke_width=2)

        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(t, scale=0.8) for t in tokens], lag_ratio=0.2),
            run_time=1.0
        )

        # Show token IDs
        ids = ["[42]", "[187]", "[93]"]
        id_texts = VGroup()
        for i, (tid, tok) in enumerate(zip(ids, tokens)):
            id_text = Text(tid, font_size=20, color=colors[i])
            id_text.next_to(tok, DOWN, buff=0.2)
            id_texts.add(id_text)

        self.play(FadeIn(id_texts, shift=UP * 0.1), run_time=0.6)
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microvit: image patches → transformer ---
    def montage_vit(self):
        label = self._montage_label("microvit.py", "01-foundations")

        # Image grid — 4x4 colored patches representing an image
        np.random.seed(99)
        patch_size = 0.6
        grid_n = 4
        image_grid = VGroup()
        patch_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
                        ACCENT_RED, ACCENT_TEAL, ACCENT_BLUE, ACCENT_GREEN,
                        ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_RED, ACCENT_TEAL,
                        ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE]
        for i in range(grid_n):
            for j in range(grid_n):
                idx = i * grid_n + j
                cell = Square(
                    side_length=patch_size,
                    fill_opacity=np.random.uniform(0.3, 0.7),
                    fill_color=patch_colors[idx],
                    stroke_width=1.5,
                    stroke_color=GREY_D
                )
                cell.move_to(LEFT * 3.5 + RIGHT * j * (patch_size + 0.05) + DOWN * i * (patch_size + 0.05))
                image_grid.add(cell)

        image_grid.move_to(LEFT * 3 + DOWN * 0.2)
        img_title = Text("Image Patches", font_size=22, color=TEXT_DIM)
        img_title.next_to(image_grid, UP, buff=0.3)

        # Arrow to transformer
        arrow = Arrow(LEFT * 0.8, RIGHT * 0.8, color=TEXT_DIM, stroke_width=2)
        arrow.move_to(ORIGIN + DOWN * 0.2)

        # Transformer stack on right
        transformer_layers = [
            ("Patch Embed", ACCENT_BLUE),
            ("+ Position", ACCENT_ORANGE),
            ("Self-Attention", ACCENT_GREEN),
            ("MLP Head", ACCENT_PURPLE),
        ]
        stack = VGroup()
        for name, color in transformer_layers:
            block = VGroup(
                RoundedRectangle(width=2.8, height=0.55, corner_radius=0.1,
                                 color=color, fill_opacity=0.2, stroke_width=2),
                Text(name, font_size=18, color=color)
            )
            stack.add(block)
        stack.arrange(DOWN, buff=0.12)
        stack.move_to(RIGHT * 3.5 + DOWN * 0.2)

        # Classification output
        cls_label = Text("[CLS] → class", font_size=20, color=ACCENT_TEAL)
        cls_label.next_to(stack, DOWN, buff=0.35)

        self.play(FadeIn(label), run_time=0.4)
        self.play(Write(img_title), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.7) for c in image_grid], lag_ratio=0.02),
            run_time=1.0
        )
        self.play(GrowArrow(arrow), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.15) for b in stack], lag_ratio=0.15),
            run_time=1.0
        )
        self.play(FadeIn(cls_label, shift=UP * 0.15), run_time=0.5)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- micromoe: expert routing ---
    def montage_moe(self):
        label = self._montage_label("micromoe.py", "02-alignment")

        # Input token
        input_box = VGroup(
            RoundedRectangle(width=1.8, height=0.7, corner_radius=0.1,
                             color=TEXT_BRIGHT, fill_opacity=0.1, stroke_width=2),
            Text("input", font_size=22, color=TEXT_BRIGHT)
        ).move_to(LEFT * 4.5)

        # Router
        router = VGroup(
            RoundedRectangle(width=1.8, height=0.9, corner_radius=0.1,
                             color=ACCENT_ORANGE, fill_opacity=0.2, stroke_width=2),
            Text("Router", font_size=22, color=ACCENT_ORANGE)
        ).move_to(LEFT * 1.5)

        # Experts
        expert_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]
        expert_labels = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
        experts = VGroup()
        for i, (name, color) in enumerate(zip(expert_labels, expert_colors)):
            exp = VGroup(
                RoundedRectangle(width=1.6, height=0.65, corner_radius=0.1,
                                 color=color, fill_opacity=0.15, stroke_width=2),
                Text(name, font_size=18, color=color)
            )
            experts.add(exp)

        experts.arrange(DOWN, buff=0.2)
        experts.move_to(RIGHT * 2)

        # Output
        output_box = VGroup(
            RoundedRectangle(width=1.8, height=0.7, corner_radius=0.1,
                             color=ACCENT_TEAL, fill_opacity=0.1, stroke_width=2),
            Text("output", font_size=22, color=ACCENT_TEAL)
        ).move_to(RIGHT * 5)

        self.play(FadeIn(label), FadeIn(input_box, shift=RIGHT * 0.3), run_time=0.5)

        # Input → Router
        a1 = Arrow(input_box.get_right(), router.get_left(), buff=0.15,
                    color=TEXT_DIM, stroke_width=2)
        self.play(FadeIn(router), GrowArrow(a1), run_time=0.5)

        # Router → Experts (top-k=2 routing: highlight 2)
        self.play(
            LaggedStart(*[FadeIn(e, shift=RIGHT * 0.2) for e in experts], lag_ratio=0.12),
            run_time=0.8
        )

        # Routing arrows — highlight top-2
        arrows_to_exp = VGroup()
        for i, exp in enumerate(experts):
            color = expert_colors[i] if i in [0, 2] else GREY_D
            width = 2.5 if i in [0, 2] else 1
            a = Arrow(router.get_right(), exp.get_left(), buff=0.15,
                      color=color, stroke_width=width)
            arrows_to_exp.add(a)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_to_exp], lag_ratio=0.08),
            run_time=0.6
        )

        # Highlight selected experts
        self.play(
            experts[0][0].animate.set_fill(opacity=0.4),
            experts[2][0].animate.set_fill(opacity=0.4),
            run_time=0.4
        )

        # Top-k label
        topk = Text("top-k = 2", font_size=20, color=ACCENT_ORANGE)
        topk.next_to(router, DOWN, buff=0.3)
        self.play(FadeIn(topk), run_time=0.4)

        # Selected → Output
        a_out1 = Arrow(experts[0].get_right(), output_box.get_left() + UP * 0.15,
                       buff=0.15, color=expert_colors[0], stroke_width=2)
        a_out2 = Arrow(experts[2].get_right(), output_box.get_left() + DOWN * 0.15,
                       buff=0.15, color=expert_colors[2], stroke_width=2)

        self.play(FadeIn(output_box), GrowArrow(a_out1), GrowArrow(a_out2), run_time=0.5)
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microspeculative: draft-verify loop ---
    def montage_speculative(self):
        label = self._montage_label("microspeculative.py", "03-systems")

        # Draft model (small)
        draft = VGroup(
            RoundedRectangle(width=2.2, height=1.0, corner_radius=0.1,
                             color=ACCENT_GREEN, fill_opacity=0.2, stroke_width=2),
            Text("Draft Model", font_size=20, color=ACCENT_GREEN),
            Text("(small, fast)", font_size=14, color=TEXT_DIM),
        )
        draft[2].next_to(draft[1], DOWN, buff=0.1)
        draft.move_to(LEFT * 3.5 + UP * 1.0)

        # Target model (large)
        target = VGroup(
            RoundedRectangle(width=2.2, height=1.0, corner_radius=0.1,
                             color=ACCENT_BLUE, fill_opacity=0.2, stroke_width=2),
            Text("Target Model", font_size=20, color=ACCENT_BLUE),
            Text("(large, accurate)", font_size=14, color=TEXT_DIM),
        )
        target[2].next_to(target[1], DOWN, buff=0.1)
        target.move_to(RIGHT * 3.5 + UP * 1.0)

        # Draft tokens
        draft_tokens = ["The", "cat", "sat", "on", "the"]
        draft_tok_group = VGroup()
        for i, tok in enumerate(draft_tokens):
            color = ACCENT_GREEN
            t = VGroup(
                RoundedRectangle(width=0.9, height=0.5, corner_radius=0.08,
                                 color=color, fill_opacity=0.2, stroke_width=1.5),
                Text(tok, font_size=16, color=color)
            )
            draft_tok_group.add(t)
        draft_tok_group.arrange(RIGHT, buff=0.1)
        draft_tok_group.move_to(DOWN * 0.3)

        # Verify checkmarks / reject
        verify_symbols = ["✓", "✓", "✓", "✗", "—"]
        verify_colors = [ACCENT_GREEN, ACCENT_GREEN, ACCENT_GREEN, ACCENT_RED, TEXT_DIM]
        verify_group = VGroup()
        for sym, color, tok in zip(verify_symbols, verify_colors, draft_tok_group):
            v = Text(sym, font_size=20, color=color)
            v.next_to(tok, DOWN, buff=0.15)
            verify_group.add(v)

        # Speedup label
        speedup = Text("~2-3x faster decoding", font_size=22, color=ACCENT_ORANGE)
        speedup.move_to(DOWN * 1.8)

        # Draft arrow
        draft_arrow = Arrow(draft.get_bottom(), draft_tok_group.get_left() + UP * 0.3,
                            buff=0.15, color=ACCENT_GREEN, stroke_width=2)
        # Verify arrow
        verify_arrow = Arrow(target.get_bottom(), draft_tok_group.get_right() + UP * 0.3,
                             buff=0.15, color=ACCENT_BLUE, stroke_width=2)

        self.play(FadeIn(label), run_time=0.4)
        self.play(FadeIn(draft, shift=RIGHT * 0.2), FadeIn(target, shift=LEFT * 0.2), run_time=0.6)

        # Draft generates tokens
        self.play(GrowArrow(draft_arrow), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(t, scale=0.8) for t in draft_tok_group], lag_ratio=0.12),
            run_time=0.8
        )

        # Target verifies
        self.play(GrowArrow(verify_arrow), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(v, shift=DOWN * 0.1) for v in verify_group], lag_ratio=0.12),
            run_time=0.8
        )

        self.play(Write(speedup), run_time=0.5)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microlstm: 4-gate architecture ---
    def montage_lstm(self):
        label = self._montage_label("microlstm.py", "01-foundations")

        # LSTM cell with 4 gates
        cell_bg = RoundedRectangle(
            width=7, height=3.5, corner_radius=0.2,
            color=ACCENT_BLUE, fill_opacity=0.05, stroke_width=2
        )
        cell_title = Text("LSTM Cell", font_size=26, color=TEXT_BRIGHT)
        cell_title.next_to(cell_bg, UP, buff=0.25)

        gates = [
            ("Forget\nGate", ACCENT_RED, "σ"),
            ("Input\nGate", ACCENT_GREEN, "σ"),
            ("Candidate\nGate", ACCENT_ORANGE, "tanh"),
            ("Output\nGate", ACCENT_PURPLE, "σ"),
        ]

        gate_group = VGroup()
        for name, color, activation in gates:
            gate = VGroup(
                RoundedRectangle(width=1.3, height=1.6, corner_radius=0.1,
                                 color=color, fill_opacity=0.2, stroke_width=2),
                Text(name, font_size=14, color=color),
                Text(activation, font_size=16, color=color, weight=BOLD),
            )
            gate[1].move_to(gate[0].get_center() + UP * 0.25)
            gate[2].move_to(gate[0].get_center() + DOWN * 0.35)
            gate_group.add(gate)

        gate_group.arrange(RIGHT, buff=0.35)
        gate_group.move_to(cell_bg.get_center())

        # Cell state arrow (horizontal line through top)
        cell_state_arrow = Arrow(LEFT * 3.8, RIGHT * 3.8, color=ACCENT_TEAL, stroke_width=2.5)
        cell_state_arrow.move_to(cell_bg.get_top() + DOWN * 0.15)
        cs_label = Text("cell state", font_size=16, color=ACCENT_TEAL)
        cs_label.next_to(cell_state_arrow, UP, buff=0.1)

        self.play(FadeIn(label), run_time=0.4)
        self.play(Write(cell_title), Create(cell_bg), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(g, scale=0.8) for g in gate_group], lag_ratio=0.15),
            run_time=1.2
        )
        self.play(GrowArrow(cell_state_arrow), FadeIn(cs_label), run_time=0.6)

        # Pulse through gates to show data flow
        for gate in gate_group:
            self.play(
                gate[0].animate.set_fill(opacity=0.5),
                run_time=0.2
            )
            self.play(
                gate[0].animate.set_fill(opacity=0.2),
                run_time=0.15
            )

        self.wait(0.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microdiffusion: noise → signal ---
    def montage_diffusion(self):
        label = self._montage_label("microdiffusion.py", "01-foundations")

        steps = 6
        np.random.seed(7)
        cell_size = 0.3
        grid_n = 8

        all_grids = []
        for step in range(steps):
            noise_level = 1.0 - step / (steps - 1)
            grid = VGroup()
            for i in range(grid_n):
                for j in range(grid_n):
                    # Blend from noise (random grey) to pattern (checkerboard-like)
                    pattern_val = ((i + j) % 2) * 0.8 + 0.1
                    noise_val = np.random.uniform(0, 1)
                    val = noise_level * noise_val + (1 - noise_level) * pattern_val

                    cell = Square(
                        side_length=cell_size,
                        fill_opacity=val,
                        fill_color=ACCENT_PURPLE,
                        stroke_width=0,
                    )
                    cell.move_to(RIGHT * j * (cell_size + 0.02) + DOWN * i * (cell_size + 0.02))
                    grid.add(cell)

            grid.move_to(ORIGIN)
            all_grids.append(grid)

        # Step labels
        step_labels = [f"t={steps - 1 - i}" for i in range(steps)]

        self.play(FadeIn(label), run_time=0.4)

        # Title
        title = Text("Denoising Process", font_size=28, color=ACCENT_PURPLE)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title), run_time=0.5)

        # Show first (noisiest) grid
        current_grid = all_grids[0]
        step_text = Text(step_labels[0], font_size=22, color=TEXT_DIM)
        step_text.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(current_grid), FadeIn(step_text), run_time=0.6)
        self.wait(0.4)

        # Animate denoising steps
        for i in range(1, steps):
            new_step_text = Text(step_labels[i], font_size=22, color=TEXT_DIM)
            new_step_text.to_edge(DOWN, buff=0.8)

            self.play(
                ReplacementTransform(current_grid, all_grids[i]),
                ReplacementTransform(step_text, new_step_text),
                run_time=0.6
            )
            current_grid = all_grids[i]
            step_text = new_step_text
            self.wait(0.15)

        # Final "clean" flash
        self.play(Indicate(current_grid, color=WHITE, scale_factor=1.05), run_time=0.5)
        self.wait(0.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- micromcts: tree search ---
    def montage_mcts(self):
        label = self._montage_label("micromcts.py", "04-agents")

        # Root node
        root = Circle(radius=0.3, color=ACCENT_ORANGE, fill_opacity=0.3, stroke_width=2)
        root_label = Text("S₀", font_size=18, color=ACCENT_ORANGE)
        root_label.move_to(root)
        root_group = VGroup(root, root_label).move_to(UP * 2.5)

        # Level 1 — 3 children
        l1_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED]
        l1_labels_text = ["S₁", "S₂", "S₃"]
        l1_nodes = VGroup()
        for i, (color, lbl) in enumerate(zip(l1_colors, l1_labels_text)):
            node = Circle(radius=0.25, color=color, fill_opacity=0.25, stroke_width=2)
            nlbl = Text(lbl, font_size=16, color=color)
            nlbl.move_to(node)
            g = VGroup(node, nlbl)
            g.move_to(LEFT * 3 + RIGHT * i * 3 + UP * 0.8)
            l1_nodes.add(g)

        # Level 2 — 2 children under S₂ (the explored path)
        l2_nodes = VGroup()
        for i, lbl in enumerate(["S₄", "S₅"]):
            node = Circle(radius=0.2, color=ACCENT_GREEN, fill_opacity=0.2, stroke_width=1.5)
            nlbl = Text(lbl, font_size=14, color=ACCENT_GREEN)
            nlbl.move_to(node)
            g = VGroup(node, nlbl)
            g.move_to(LEFT * 1 + RIGHT * i * 2 + DOWN * 0.8)
            l2_nodes.add(g)

        # Edges
        edges_l1 = VGroup()
        for child in l1_nodes:
            edge = Line(root_group.get_bottom(), child.get_top(), color=GREY_D, stroke_width=1.5)
            edges_l1.add(edge)

        edges_l2 = VGroup()
        for child in l2_nodes:
            edge = Line(l1_nodes[1].get_bottom(), child.get_top(), color=ACCENT_GREEN, stroke_width=1.5)
            edges_l2.add(edge)

        # MCTS phases
        phases = VGroup(
            Text("1. Select", font_size=18, color=ACCENT_BLUE),
            Text("2. Expand", font_size=18, color=ACCENT_GREEN),
            Text("3. Simulate", font_size=18, color=ACCENT_ORANGE),
            Text("4. Backpropagate", font_size=18, color=ACCENT_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        phases.move_to(RIGHT * 4.5 + DOWN * 0.5)

        # UCB formula
        ucb = Text("UCB = w/n + c√(ln N / n)", font_size=18, color=TEXT_DIM)
        ucb.move_to(DOWN * 2.2)

        self.play(FadeIn(label), run_time=0.4)
        self.play(FadeIn(root_group, scale=0.8), run_time=0.4)
        self.play(
            LaggedStart(*[Create(e) for e in edges_l1], lag_ratio=0.15),
            run_time=0.5
        )
        self.play(
            LaggedStart(*[FadeIn(n, scale=0.8) for n in l1_nodes], lag_ratio=0.15),
            run_time=0.6
        )

        # Highlight selection path to S₂
        self.play(
            l1_nodes[1][0].animate.set_fill(opacity=0.5),
            edges_l1[1].animate.set_color(ACCENT_GREEN),
            run_time=0.4
        )

        # Expand S₂
        self.play(
            LaggedStart(*[Create(e) for e in edges_l2], lag_ratio=0.15),
            run_time=0.4
        )
        self.play(
            LaggedStart(*[FadeIn(n, scale=0.8) for n in l2_nodes], lag_ratio=0.15),
            run_time=0.5
        )

        # Show phases and UCB
        self.play(
            LaggedStart(*[FadeIn(p, shift=RIGHT * 0.15) for p in phases], lag_ratio=0.12),
            run_time=0.8
        )
        self.play(FadeIn(ucb, shift=UP * 0.1), run_time=0.4)
        self.wait(0.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microgpt: training loop + loss ---
    def montage_gpt(self):
        label = self._montage_label("microgpt.py", "01-foundations")

        # Architecture diagram (simplified)
        layers = [
            ("Embedding", ACCENT_BLUE),
            ("Self-Attention", ACCENT_ORANGE),
            ("Feed-Forward", ACCENT_GREEN),
            ("Softmax", ACCENT_PURPLE),
        ]

        arch = VGroup()
        for name, color in layers:
            block = VGroup(
                RoundedRectangle(width=3, height=0.6, corner_radius=0.1,
                                 color=color, fill_opacity=0.2, stroke_width=2),
                Text(name, font_size=20, color=color)
            )
            arch.add(block)

        arch.arrange(DOWN, buff=0.15)
        arch.move_to(LEFT * 3.5 + DOWN * 0.2)

        arch_title = Text("Transformer", font_size=24, color=TEXT_BRIGHT)
        arch_title.next_to(arch, UP, buff=0.3)

        # Loss curve on right
        axes = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=2.8,
            axis_config={"color": GREY_D, "stroke_width": 1, "include_ticks": False},
        ).move_to(RIGHT * 3 + DOWN * 0.2)

        x_label = Text("epoch", font_size=16, color=TEXT_DIM)
        x_label.next_to(axes, DOWN, buff=0.15)
        y_label = Text("loss", font_size=16, color=TEXT_DIM)
        y_label.next_to(axes, LEFT, buff=0.15)

        # Exponential decay loss curve
        loss_curve = axes.plot(
            lambda x: 3.5 * np.exp(-0.08 * x) + 0.3,
            x_range=[0, 50],
            color=ACCENT_RED,
            stroke_width=2.5
        )

        loss_title = Text("Training Loss", font_size=22, color=ACCENT_RED)
        loss_title.next_to(axes, UP, buff=0.3)

        self.play(FadeIn(label), run_time=0.4)

        # Build architecture
        self.play(Write(arch_title), run_time=0.4)
        arrows_arch = VGroup()
        for i, block in enumerate(arch):
            self.play(FadeIn(block, shift=DOWN * 0.2), run_time=0.3)
            if i < len(arch) - 1:
                a = Arrow(
                    arch[i].get_bottom(), arch[i + 1].get_top(),
                    buff=0.05, color=GREY_D, stroke_width=1.5
                )
                arrows_arch.add(a)
                self.play(GrowArrow(a), run_time=0.15)

        # Draw loss curve
        self.play(Write(loss_title), run_time=0.4)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.5)
        self.play(Create(loss_curve), run_time=1.8)

        # Generated text sample
        gen_text = Text('"Aelira\nKarthen\nZylox"', font_size=20, color=ACCENT_TEAL)
        gen_text.next_to(axes, DOWN, buff=0.6)
        gen_label = Text("Generated names ↑", font_size=16, color=TEXT_DIM)
        gen_label.next_to(gen_text, DOWN, buff=0.15)

        self.play(FadeIn(gen_text, shift=UP * 0.2), FadeIn(gen_label), run_time=0.6)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # =================================================================
    # ACT 3: Repository Structure + Stats (47–60s)
    # =================================================================
    def act3_structure(self):
        # Section title
        section = Text("41 scripts  ·  4 tiers  ·  zero dependencies", font_size=30, color=TEXT_BRIGHT)
        section.to_edge(UP, buff=0.5)
        self.play(Write(section), run_time=1.0)

        # Four tier columns
        tiers = [
            ("01-foundations", "14 scripts", ACCENT_BLUE, [
                "microgpt", "micrornn", "microlstm",
                "microtokenizer", "microembedding",
                "microrag", "microdiffusion", "microvae",
                "microbert", "microconv", "microresnet",
                "microvit", "microgan", "microoptimizer",
            ]),
            ("02-alignment", "10 scripts", ACCENT_GREEN, [
                "microlora", "microdpo", "microppo",
                "micromoe", "microgrpo", "microreinforce",
                "microqlora", "microbatchnorm",
                "microdropout", "adam_vs_sgd",
            ]),
            ("03-systems", "13 scripts", ACCENT_ORANGE, [
                "microattention", "microkv", "microquant",
                "microflash", "microbeam", "microrope",
                "microssm", "micropaged", "microparallel",
                "microcheckpoint", "microspeculative",
                "microbm25", "microvectorsearch",
            ]),
            ("04-agents", "2 scripts", ACCENT_PURPLE, [
                "micromcts", "microreact",
            ]),
        ]

        columns = VGroup()
        for tier_name, count, color, scripts in tiers:
            # Header
            header = VGroup(
                Text(tier_name, font_size=22, weight=BOLD, color=color),
                Text(count, font_size=16, color=TEXT_DIM),
            ).arrange(DOWN, buff=0.12)

            # Script list
            script_list = VGroup()
            for s in scripts:
                t = Text(s + ".py", font_size=11, color=color)
                t.set_opacity(0.75)
                script_list.add(t)
            script_list.arrange(DOWN, buff=0.06, aligned_edge=LEFT)

            col = VGroup(header, script_list).arrange(DOWN, buff=0.25)
            columns.add(col)

        columns.arrange(RIGHT, buff=0.7, aligned_edge=UP)
        columns.move_to(DOWN * 0.3)

        # Animate columns appearing
        for col in columns:
            header, scripts = col[0], col[1]
            self.play(FadeIn(header, shift=UP * 0.2), run_time=0.4)
            self.play(
                LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in scripts], lag_ratio=0.03),
                run_time=0.6
            )

        self.wait(0.6)

        # Key stats bar at bottom — includes repo metrics
        stats = VGroup(
            Text("★ 500+", font_size=24, weight=BOLD, color=ACCENT_ORANGE),
            Text("·", font_size=22, color=GREY_D),
            Text("55 forks", font_size=22, color=TEXT_BRIGHT),
            Text("·", font_size=22, color=GREY_D),
            Text("pure Python", font_size=22, color=ACCENT_TEAL),
            Text("·", font_size=22, color=GREY_D),
            Text("zero pip install", font_size=22, color=ACCENT_TEAL),
        ).arrange(RIGHT, buff=0.3)
        stats.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(stats, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)
        self.wait(0.3)

    # =================================================================
    # ACT 4: CTA + GitHub URL (60–70s)
    # =================================================================
    def act4_cta(self):
        cta = Text("Clone and run in 30 seconds", font_size=36, color=TEXT_BRIGHT, weight=BOLD)
        cta.move_to(UP * 1.5)

        # Terminal-style command
        terminal_bg = RoundedRectangle(
            width=10, height=1.8, corner_radius=0.2,
            color="#161b22", fill_opacity=0.9, stroke_width=1, stroke_color=GREY_D
        )
        terminal_bg.move_to(DOWN * 0.2)

        cmd1 = Text("$ git clone github.com/Mathews-Tom/no-magic", font_size=20, color=ACCENT_GREEN)
        cmd2 = Text("$ python 01-foundations/microgpt.py", font_size=20, color=ACCENT_GREEN)
        cmds = VGroup(cmd1, cmd2).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        cmds.move_to(terminal_bg)

        url = Text(
            "github.com/Mathews-Tom/no-magic",
            font_size=32, weight=BOLD, color=ACCENT_BLUE
        )
        url.move_to(DOWN * 2)

        # Star icon substitute
        star = Text("★", font_size=28, color=ACCENT_ORANGE)
        star.next_to(url, RIGHT, buff=0.4)

        self.play(Write(cta), run_time=1.0)
        self.play(FadeIn(terminal_bg), run_time=0.4)
        self.play(Write(cmd1), run_time=1.0)
        self.play(Write(cmd2), run_time=0.8)
        self.wait(0.6)

        self.play(FadeIn(url, shift=UP * 0.2), FadeIn(star, scale=1.5), run_time=0.7)

        # Hold for screenshot
        self.wait(4.5)

    # =================================================================
    # Helpers
    # =================================================================
    def _montage_label(self, filename: str, tier: str) -> VGroup:
        """Top-right label showing current script name and tier."""
        name = Text(filename, font_size=20, weight=BOLD, color=TEXT_BRIGHT)
        tier_text = Text(tier, font_size=16, color=TEXT_DIM)
        group = VGroup(name, tier_text).arrange(DOWN, buff=0.1, aligned_edge=RIGHT)
        group.to_corner(UR, buff=0.4)
        return group
