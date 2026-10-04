from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assignment1" / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY = "#18324A"
BLUE = "#2F6B9A"
TEAL = "#287D75"
RED = "#B64A4A"
INK = "#24313B"
MUTED = "#64727E"
GRID = "#D8E0E6"
PALE = "#F3F6F8"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def centered(draw, box, text, text_font, fill=INK):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=8, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=text_font,
        fill=fill,
        spacing=8,
        align="center",
    )


def arrow(draw, start, end, fill=BLUE, width=5):
    draw.line([start, end], fill=fill, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 9), (x - 16, y + 9)], fill=fill)


def architecture():
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "广义五子棋系统结构", font=font(48, True), fill=NAVY)
    draw.text((92, 120), "统一规则状态贯穿界面、AI 协议与实验", font=font(25), fill=MUTED)

    boxes = [
        ((90, 220, 450, 390), "GUI 与三种模式\n参数设置  外部 AI 加载", "#EAF2F8"),
        ((620, 220, 980, 390), "GomokuGame\n合法性  状态  胜负  和棋", "#E9F5F2"),
        ((1150, 220, 1510, 390), "统一 AI 协议\nget_move 与时限回退", "#F8EFEC"),
        ((300, 570, 660, 740), "四种搜索策略\nBaseline  Enhanced  MCTS  Hybrid", "#F0F3F7"),
        ((940, 570, 1300, 740), "实验与测试\n1500 局主矩阵\n战术基准  单元测试", "#F0F3F7"),
    ]
    for box, label, fill in boxes:
        draw.rounded_rectangle(box, radius=16, fill=fill, outline=GRID, width=3)
        size = 24 if "实验与测试" in label else 27
        centered(draw, box, label, font(size, "GomokuGame" in label))

    arrow(draw, (450, 305), (620, 305))
    arrow(draw, (980, 305), (1150, 305))
    arrow(draw, (1330, 390), (1170, 570))
    arrow(draw, (450, 570), (720, 390))
    arrow(draw, (660, 655), (940, 655))
    draw.text((90, 830), "设计原则  规则单一来源  输入只读  配置参数化  实验可追溯", font=font(25), fill=MUTED)
    image.save(OUT / "system_architecture.png", quality=95)


def line_chart():
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "Hybrid 对 Enhanced 的胜率", font=font(48, True), fill=NAVY)
    draw.text((92, 120), "每个点 50 局  颜色交换  虚线表示 50%", font=font(24), fill=MUTED)

    left, top, right, bottom = 170, 210, 1480, 730
    for value in range(0, 81, 10):
        y = bottom - (value / 80) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((95, y - 15), f"{value}%", font=font(20), fill=MUTED)
    y50 = bottom - (50 / 80) * (bottom - top)
    for x in range(left, right, 24):
        draw.line((x, y50, min(x + 12, right), y50), fill="#7D8992", width=3)

    xs = [360, 800, 1240]
    labels = ["0.5 秒", "1.0 秒", "5.0 秒"]
    for x, label in zip(xs, labels):
        draw.text((x - 48, bottom + 28), label, font=font(22), fill=INK)

    series = [
        ("9x9 K4", [50, 48, 50], BLUE),
        ("15x15 K5", [30, 40, 56], RED),
    ]
    for name, values, color in series:
        points = []
        for x, value in zip(xs, values):
            y = bottom - (value / 80) * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=7)
        for point_index, ((x, y), value) in enumerate(zip(points, values)):
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color, outline="white", width=3)
            label_y = y - 48
            if name == "9x9 K4" and point_index == 2:
                label_y = y + 18
            draw.text((x - 24, label_y), f"{value}%", font=font(22, True), fill=color)

    draw.line((950, 820, 1020, 820), fill=BLUE, width=7)
    draw.text((1040, 803), "9x9 K4", font=font(22), fill=INK)
    draw.line((1220, 820, 1290, 820), fill=RED, width=7)
    draw.text((1310, 803), "15x15 K5", font=font(22), fill=INK)
    image.save(OUT / "hybrid_vs_enhanced.png", quality=95)


def aggregate_chart():
    image = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "对 Enhanced 的聚合成绩", font=font(46, True), fill=NAVY)
    draw.text((92, 115), "每种算法 300 局  数值为积分率", font=font(24), fill=MUTED)

    left, right = 430, 1370
    names = ["Baseline Alpha-Beta", "MCTS + UCT", "Hybrid Threat Search"]
    values = [16.3, 33.7, 45.7]
    colors = ["#8A98A5", TEAL, BLUE]
    for index, (name, value, color) in enumerate(zip(names, values, colors)):
        y = 245 + index * 155
        draw.text((90, y + 16), name, font=font(24, index == 2), fill=INK)
        draw.rounded_rectangle((left, y, right, y + 72), radius=12, fill=PALE)
        x = left + int((right - left) * value / 60)
        draw.rounded_rectangle((left, y, x, y + 72), radius=12, fill=color)
        draw.text((x + 18, y + 15), f"{value:.1f}%", font=font(26, True), fill=color)

    draw.text((90, 690), "积分率 = (胜局 + 0.5 x 和棋) / 总局数", font=font(22), fill=MUTED)
    image.save(OUT / "aggregate_against_enhanced.png", quality=95)


if __name__ == "__main__":
    architecture()
    line_chart()
    aggregate_chart()
