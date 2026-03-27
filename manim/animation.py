from manim import *
import numpy as np
import random


class NeuralNetworkAnimation(Scene):
    def construct(self):
        layer_sizes = [3, 5, 4, 2]
        neuron_radius = 0.2

        # 1. Create Nodes (Neurons)
        layers = VGroup()
        for size in layer_sizes:
            layer = VGroup(
                *[
                    Circle(radius=neuron_radius, color=BLUE, fill_opacity=0.8)
                    for _ in range(size)
                ]
            )
            layer.arrange(DOWN, buff=0.4)
            layers.add(layer)

        layers.arrange(RIGHT, buff=2)

        # 2. Create Edges
        edge_layers = VGroup()

        for i in range(len(layers) - 1):
            layer_edges = VGroup()
            for n1 in layers[i]:
                for n2 in layers[i + 1]:
                    # Get center coordinates
                    c1 = n1.get_center()
                    c2 = n2.get_center()

                    # Calculate the direction vector between the two centers
                    vector = c2 - c1
                    direction = vector / np.linalg.norm(vector)

                    # Offset the start and end points by the radius of the circles
                    start_pt = c1 + direction * neuron_radius
                    end_pt = c2 - direction * neuron_radius

                    edge = Line(start_pt, end_pt, stroke_width=1.5, color=DARK_GRAY)
                    layer_edges.add(edge)
            edge_layers.add(layer_edges)

        # 3. Create Labels
        labels = VGroup(
            Text("Input Layer", font_size=28).next_to(layers[0], UP, buff=0.5),
            Text("Hidden Layer", font_size=28).next_to(
                VGroup(*layers[1:-1]), UP, buff=0.5
            ),
            Text("Output Layer", font_size=28).next_to(layers[-1], UP, buff=0.5),
        )

        # --- Animations ---

        self.play(Write(labels))
        self.play(
            LaggedStart(
                *[FadeIn(layer, shift=UP * 0.5) for layer in layers], lag_ratio=0.3
            )
        )
        self.wait(2)

        self.play(
            LaggedStart(
                *[Create(edge_group) for edge_group in edge_layers], lag_ratio=0.5
            ),
            run_time=2,
        )
        self.wait(2)

        # Simulate Forward Pass
        for i in range(len(layers) - 1):
            self.play(layers[i].animate.set_color(YELLOW), run_time=0.3)
            self.play(
                edge_layers[i].animate.set_color(YELLOW).set_stroke(width=3),
                run_time=0.5,
            )
            self.play(
                edge_layers[i].animate.set_color(DARK_GRAY).set_stroke(width=1.5),
                layers[i].animate.set_color(BLUE),
                run_time=0.3,
            )

        self.play(layers[-1].animate.set_color(GREEN), run_time=0.5)
        self.wait(2)


class LinearClassifierIntro(Scene):
    def construct(self):
        # context and backgound
        intro_text = Text(
            "Task: Identify all the cats from 10000 images", font_size=36
        ).to_edge(UP)
        self.play(Write(intro_text))
        wait_time = 3

        # mimic 10000 images in pixels
        grid = VGroup(
            *[
                Square(
                    side_length=0.15,
                    fill_opacity=0.8,
                    color=rgb_to_color(
                        [random.random(), random.random(), random.random()]
                    ),
                ).set_stroke(width=0.2)
                for _ in range(900)
            ]
        )

        grid.arrange_in_grid(rows=30, cols=30, buff=0.05)
        grid.scale_to_fit_height(6)
        grid.move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(s) for s in grid], lag_ratio=0.002, run_time=2))
        self.wait(1)

        # Focus on single image
        self.play(FadeOut(grid, intro_text))

        # load image
        try:
            cat_img = ImageMobject("images/cat.jpg").scale(0.3).shift(LEFT * 3)
        except:
            # use colorful rectangle for place holder
            cat_img = Rectangle(color=ORANGE, fill_opacity=1, width=3, height=3).shift(
                LEFT * 3
            )
            cat_label = Text("Cat Image", color=WHITE).scale(0.5).move_to(cat_img)
            self.add(cat_label)

        human_view = (
            Text("Human View: cute cat", color=YELLOW).scale(0.6).next_to(cat_img, UP)
        )
        self.play(FadeIn(cat_img), Write(human_view))
        self.wait(1)

        # Scanning
        scan_line = Line(
            cat_img.get_left(), cat_img.get_right(), color=GREEN
        ).set_stroke(width=10)
        scan_line.set_y(cat_img.get_top()[1])
        self.play(scan_line.animate.shift(DOWN * 3), run_time=2, rate_func=linear)
        self.play(FadeOut(scan_line))
        # RGB matrix
        computer_view = (
            Text("Computer View: numbers", color=BLUE)
            .scale(0.6)
            .next_to(human_view, RIGHT, buff=1)
        )

        # a random matrix for pixel value
        matrix_rows = []
        for _ in range(5):
            row = []
            for _ in range(5):
                rgb_val = np.random.randint(0, 255, size=3)
                v = MathTex(
                    f"\\begin{{bmatrix}} {rgb_val[0]} \\\\ {rgb_val[1]} \\\\ {rgb_val[2]} \\end{{bmatrix}}",
                    font_size=15,
                )
                row.append(v)
            matrix_rows.append(row)
        pixel_vector_group = VGroup(
            *[item for sublist in matrix_rows for item in sublist]
        )
        pixel_vector_group.arrange_in_grid(rows=5, cols=5, buff=0.1).next_to(
            computer_view, DOWN
        )

        self.play(Write(computer_view))
        self.play(FadeIn(pixel_vector_group))

        classifier_text = Text("Linear Classifier", font_size=18).next_to(
            pixel_vector_group, DOWN
        )

        self.play(
            Write(classifier_text),
        )

        result_label = (
            Text("Deciding...", color=GREY).scale(0.7).next_to(classifier_text, DOWN)
        )
        self.play(Write(result_label))

        final_res = Text("It's a CAT!", color=GREEN).scale(0.8).move_to(result_label)
        self.play(Transform(result_label, final_res))

        self.wait(2)
