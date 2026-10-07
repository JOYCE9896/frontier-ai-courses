from manim import *

FONT = "Songti SC"
config.background_color = "#1c1c1e"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def cell(label, color, filled):
    sq = Square(side_length=0.62, color=color, stroke_width=2)
    if filled:
        sq.set_fill(color, opacity=0.55)
    return VGroup(sq, T(label, 20))


class KVCache(Scene):
    def construct(self):
        title = T("逐个生成 token 时，每一步要算多少 K、V", 34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        words = ["今天", "天气", "很", "好"]
        rows_y = [1.15, -1.05]
        heads = [T("没有 KV 缓存", 26, RED_C), T("有 KV 缓存", 26, GREEN_C)]
        for h, y in zip(heads, rows_y):
            h.move_to([-4.6, y + 0.9, 0])
        self.play(FadeIn(heads[0]), FadeIn(heads[1]))

        legend = VGroup(
            VGroup(Square(0.3, color=RED_C).set_fill(RED_C, 0.55), T("这一步重新计算", 22)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.3, color=GREEN_C).set_fill(GREEN_C, 0.55), T("这一步新算", 22)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.3, color=GREY_B), T("从缓存读取", 22)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(legend))

        x0 = -4.2
        dx = 0.85
        top_cells = VGroup()
        bot_cells = VGroup()
        counts = [None, None]
        for step in range(1, 5):
            new_top = VGroup()
            new_bot = VGroup()
            for i in range(step):
                c = cell(words[i], RED_C if True else GREY_B, True)
                c.move_to([x0 + i * dx, rows_y[0], 0])
                new_top.add(c)
                is_new = i == step - 1
                b = cell(words[i], GREEN_C if is_new else GREY_B, is_new)
                b.move_to([x0 + i * dx, rows_y[1], 0])
                new_bot.add(b)
            ct = T(f"算了 {step} 个位置的 K、V", 24, RED_C).move_to([3.3, rows_y[0], 0])
            cb = T(f"算了 1 个位置的 K、V", 24, GREEN_C).move_to([3.3, rows_y[1], 0])
            step_lab = T(f"第 {step} 步：新 token 是「{words[step-1]}」", 26, GREY_A).next_to(title, DOWN, buff=0.25)
            anims = [ReplacementTransform(top_cells, new_top) if len(top_cells) else FadeIn(new_top),
                     ReplacementTransform(bot_cells, new_bot) if len(bot_cells) else FadeIn(new_bot)]
            if counts[0] is None:
                anims += [FadeIn(ct), FadeIn(cb), FadeIn(step_lab)]
            else:
                anims += [ReplacementTransform(counts[0], ct), ReplacementTransform(counts[1], cb),
                          ReplacementTransform(self.step_lab, step_lab)]
            self.play(*anims, run_time=0.9)
            self.step_lab = step_lab
            counts = [ct, cb]
            top_cells, bot_cells = new_top, new_bot
            self.wait(0.6)

        total = VGroup(
            T("生成 n 个 token：没有缓存共算约 n²/2 次，有缓存只算 n 次", 26, YELLOW),
            T("代价是要把所有位置的 K、V 一直存在显存里", 24, GREY_A),
        ).arrange(DOWN, buff=0.2).next_to(legend, UP, buff=0.35)
        self.play(Write(total[0]), run_time=1.5)
        self.play(FadeIn(total[1]))
        self.wait(3)
