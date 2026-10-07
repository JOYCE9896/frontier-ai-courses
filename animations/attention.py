from manim import *

FONT = "Songti SC"
config.background_color = "#1c1c1e"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class Attention(Scene):
    def construct(self):
        title = T("位置 3 怎样从前文取信息", 36).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)

        keys = ["(1, 0)", "(0, 1)", "(1, 1)"]
        vals = ["(1, 0)", "(0, 2)", "(3, 1)"]
        cols = VGroup()
        for i in range(3):
            box = RoundedRectangle(width=2.6, height=1.7, corner_radius=0.12, color=GREY_B)
            name = T(f"位置 {i+1}", 26)
            k = T(f"键 {keys[i]}", 24, BLUE_C)
            v = T(f"值 {vals[i]}", 24, GREEN_C)
            inner = VGroup(name, k, v).arrange(DOWN, buff=0.15)
            cols.add(VGroup(box, inner))
        cols.arrange(RIGHT, buff=0.7).shift(UP * 0.9)
        for c in cols:
            c[1].move_to(c[0])
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in cols], lag_ratio=0.25), run_time=1.5)

        q = T("位置 3 的查询 q = (2, 0)", 28, YELLOW).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(q, shift=UP * 0.2))
        self.wait(0.5)

        # step 1: dot products
        step = T("① 查询和每个键做点积", 26, GREY_A).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(step))
        dots = VGroup(*[T(f"q·k = {s}", 26, YELLOW) for s in ["2", "0", "2"]])
        for d, c in zip(dots, cols):
            d.next_to(c, DOWN, buff=0.25)
        self.play(LaggedStart(*[Write(d) for d in dots], lag_ratio=0.3), run_time=1.5)
        self.wait(0.6)

        # step 2: scale
        step2 = T("② 除以 √2（向量维度的平方根）", 26, GREY_A).to_edge(DOWN, buff=0.4)
        scaled = VGroup(*[T(s, 26, YELLOW) for s in ["1.41", "0", "1.41"]])
        for s, d in zip(scaled, dots):
            s.move_to(d)
        self.play(ReplacementTransform(step, step2), *[ReplacementTransform(d, s) for d, s in zip(dots, scaled)])
        self.wait(0.6)

        # step 3: softmax -> bars
        step3 = T("③ softmax：变成加起来等于 1 的权重", 26, GREY_A).to_edge(DOWN, buff=0.4)
        weights = [0.446, 0.108, 0.446]
        bars = VGroup()
        labels = VGroup()
        base_y = scaled[0].get_bottom()[1] - 1.65
        for w, s in zip(weights, scaled):
            bar = Rectangle(width=0.9, height=w * 2.7, fill_color=YELLOW, fill_opacity=0.85, stroke_width=0)
            bar.move_to([s.get_center()[0], base_y + w * 1.35, 0])
            bars.add(bar)
            lab = T(f"{w:.3f}", 24, YELLOW).next_to(bar, UP, buff=0.1)
            labels.add(lab)
        self.play(ReplacementTransform(step2, step3),
                  *[ReplacementTransform(s, l) for s, l in zip(scaled, labels)],
                  *[GrowFromEdge(b, DOWN) for b in bars], run_time=1.5)
        self.wait(1)

        # step 4: weighted sum of values
        step4 = T("④ 按权重把各位置的值加起来", 26, GREY_A).to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(step3, step4))
        for c, w in zip(cols, weights):
            self.play(c[1][2].animate.set_opacity(0.25 + w * 1.6), run_time=0.3)
        out = T("输出 ≈ 0.446×(1,0) + 0.108×(0,2) + 0.446×(3,1) = (1.78, 0.66)", 24, GREEN_C)
        out.next_to(bars, DOWN, buff=0.35)
        self.play(Write(out), run_time=2)
        self.wait(0.8)
        note = T("位置 1 和 3 的键与查询方向一致，分到了大部分权重", 24, GREY_A).to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(step4, note))
        self.wait(2.5)
