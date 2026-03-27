from manim import *
import numpy as np
import random


CAT_IMAGE_PATH = "./images/cat.jpg"


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
            cat_img = ImageMobject(CAT_IMAGE_PATH).scale(0.3).shift(LEFT * 3)
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


class KNN2DScene(Scene):
    def construct(self):
        # 1. Setup Plane and Points
        axes = Axes(
            x_range=[-5, 5], y_range=[-3, 3], axis_config={"include_tip": False}
        )

        # Generate random points (Red and Blue)
        np.random.seed(42)
        # Class 1 (Red)
        red_cluster = np.array([-2, 1, 0])
        red_points = [
            red_cluster
            + np.array([np.random.normal(0, 0.8), np.random.normal(0, 0.8), 0])
            for _ in range(8)
        ]
        # Class 2 (Blue)
        blue_cluster = np.array([2, -1, 0])
        blue_points = [
            blue_cluster
            + np.array([np.random.normal(0, 0.8), np.random.normal(0, 0.8), 0])
            for _ in range(8)
        ]

        red_dots = VGroup(*[Dot(point=p, color=RED) for p in red_points])
        blue_dots = VGroup(*[Dot(point=p, color=BLUE) for p in blue_points])

        # Target point (Green dashed outline)
        target_pos = np.array([0, 0, 0])
        target_dot = Dot(point=target_pos, color=GREEN)
        target_outline = DashedVMobject(Circle(radius=0.3, color=GREEN)).move_to(
            target_pos
        )

        # Adding elements
        self.add(axes)
        self.play(FadeIn(red_dots), FadeIn(blue_dots))
        self.play(Create(target_dot), Create(target_outline))

        # Problem statement (implicit in visual)
        text_prob = Text("Classify this point", font_size=24).next_to(
            target_outline, UP
        )
        self.play(Write(text_prob))
        self.wait(2)

        # 2. Find k=5 Neighbors
        all_points = [(p, RED) for p in red_points] + [(p, BLUE) for p in blue_points]
        all_points.sort(key=lambda x: np.linalg.norm(x[0] - target_pos))
        k = 5
        neighbors = all_points[:k]

        lines = VGroup()
        for p, color in neighbors:
            lines.add(Line(target_pos, p, stroke_width=2, color=GRAY_A))

        self.play(FadeOut(text_prob))
        self.play(Create(lines))

        # Highlight neighbors
        highlight_circles = VGroup(
            *[Circle(radius=0.2, color=YELLOW).move_to(p[0]) for p in neighbors]
        )
        self.play(Create(highlight_circles))

        # 3. Voting Label
        count_red = sum(1 for p, color in neighbors if color == RED)
        count_blue = sum(1 for p, color in neighbors if color == BLUE)

        label_text = f"k={k}  {count_red} Red, {count_blue} Blue"
        label = Text(label_text, font_size=24).to_edge(UP)

        if count_red > count_blue:
            result_color = RED
            result_label = "RED"
        else:
            result_color = BLUE
            result_label = "BLUE"

        result_text = Text(
            f"Result: {result_label}", font_size=32, color=result_color
        ).next_to(label, DOWN)

        self.play(Write(label))
        self.wait(1)
        self.play(target_dot.animate.set_color(result_color))
        self.play(Write(result_text))
        self.wait(3)


class KNNImageSpace(Scene):
    def construct(self):
        # Title
        title = Text("CIFAR-10 kNN: High-Dimensional Space", font_size=32).to_edge(UP)
        self.play(Write(title))

        # 1. Image to Vector Concept
        cat_img = (
            ImageMobject(CAT_IMAGE_PATH).scale_to_fit_height(2).to_edge(LEFT, buff=1)
        )
        img_label = Text("32x32x3 Cat Image", font_size=20).next_to(cat_img, DOWN)

        img_group = Group(cat_img, img_label)

        # Vector representation (simplified vector)
        vector_box = Rectangle(height=0.5, width=4, color=WHITE).to_edge(RIGHT, buff=1)
        vector_text = Text("[128, 45, ..., 210]", font_size=18).move_to(
            vector_box.get_center()
        )
        vector_label = MathTex(r"I_i \in \mathbb{R}^{3072}", font_size=30).next_to(
            vector_box, DOWN
        )
        vec_group = VGroup(vector_box, vector_text, vector_label)

        arrow = Arrow(
            start=cat_img.get_right(), end=vector_box.get_left(), color=YELLOW
        )

        # Animate vectorization
        self.play(FadeIn(img_group))
        self.play(GrowArrow(arrow))
        self.play(FadeIn(vec_group))
        self.wait(2)

        # 2. The Abstract High-D Space
        self.play(FadeOut(img_group), FadeOut(arrow), FadeOut(vec_group))

        axes_label = Tex("Conceptual 3072D Space", font_size=20).to_edge(DOWN)
        self.play(Write(axes_label))

        # Clusters of 10 CIFAR-10 classes
        clusters = VGroup()
        class_colors = [
            RED,
            BLUE,
            GREEN,
            YELLOW,
            PURPLE,
            ORANGE,
            PINK,
            WHITE,
            TEAL,
            GOLD,
        ]
        for color in class_colors:
            center = np.array([np.random.uniform(-4, 4), np.random.uniform(-3, 3), 0])
            cluster = VGroup(
                *[
                    Dot(
                        point=center + np.random.normal(0, 0.4, 3),
                        color=color,
                        radius=0.03,
                    )
                    for _ in range(20)
                ]
            )
            clusters.add(cluster)

        clusters.scale(0.6)
        clusters.move_to(ORIGIN)

        self.play(FadeIn(clusters))
        self.wait(2)

        # 3. kNN Query (Cat Image as a new point)
        # We need a point near the "Cat Cluster" (Assume RED is Cat cluster)
        cat_cluster = clusters[0]
        query_point_pos = cat_cluster[0].get_center() + np.array([0.5, 0.5, 0])
        query_point = Dot(point=query_point_pos, color=GRAY, radius=0.1)

        mini_cat_img = (
            ImageMobject(CAT_IMAGE_PATH)
            .scale_to_fit_height(0.6)
            .move_to(query_point_pos + UP * 0.5)
        )
        self.play(FadeIn(query_point), FadeIn(mini_cat_img))

        # Visualizing Distance (Connecting to a few neighbors)
        k_images = 5

        # Calculate distances to ALL points in the clusters
        all_points = []
        for cluster in clusters:
            for point_dot in cluster:
                dist = np.linalg.norm(query_point.get_center() - point_dot.get_center())
                all_points.append((point_dot, dist))

        all_points.sort(key=lambda x: x[1])  # Sort by distance
        neighbors = all_points[:k_images]

        neighbor_lines = VGroup()
        neighbor_circles = VGroup()

        for dot, dist in neighbors:
            line = Line(
                query_point.get_center(),
                dot.get_center(),
                stroke_width=1.5,
                color=GRAY_A,
            )
            circle = Circle(radius=0.15, color=YELLOW).move_to(dot.get_center())
            neighbor_lines.add(line)
            neighbor_circles.add(circle)

        self.play(Create(neighbor_lines))
        self.play(Create(neighbor_circles))

        # Text explanation of Euclidean Distance
        dist_formula = MathTex(
            r"d(I_{query}, I_{neigh}) = \sqrt{\sum (I_{q, p}-I_{n,p})^2}", font_size=30
        ).to_edge(UP, buff=1.2)
        self.play(Write(dist_formula))
        self.wait(2)

        # 4. Result / Label Assignment
        self.play(query_point.animate.set_color(RED))
        final_text = Text("Predicted Label: CAT", color=RED, font_size=36).next_to(
            mini_cat_img, RIGHT, buff=0.5
        )
        self.play(Write(final_text))
        self.wait(4)


class LinearClassifier(Scene):
    def construct(self):
        # --- Part 1: Matrix Multiplication ---
        title = Text("Linear Classifier: Wx = s", font_size=32).to_edge(UP)
        self.add(title)

        # 1. Image to Vector Concept
        cat_img = (
            ImageMobject(CAT_IMAGE_PATH).scale_to_fit_height(2).to_edge(LEFT, buff=1)
        )
        img_label = Text("32x32x3 Cat Image", font_size=20).next_to(cat_img, DOWN)

        img_group = Group(cat_img, img_label)

        # Vector representation (simplified vector)
        vector_box = Rectangle(height=0.5, width=4, color=WHITE).to_edge(RIGHT, buff=1)
        vector_text = Text("[128, 45, ..., 210]", font_size=18).move_to(
            vector_box.get_center()
        )
        vector_label = Tex(r"a 3072-dimension vector x ", font_size=30).next_to(
            vector_box, DOWN
        )
        vec_group = VGroup(vector_box, vector_text, vector_label)

        arrow = Arrow(
            start=cat_img.get_right(), end=vector_box.get_left(), color=YELLOW
        )

        # Animate vectorization
        self.play(FadeIn(img_group))
        self.play(GrowArrow(arrow))
        self.play(FadeIn(vec_group))
        self.wait(2)

        self.play(FadeOut(img_group), FadeOut(arrow), FadeOut(vec_group))

        # Matrix W (Placed on the left first)
        rows_w = (
            VGroup(
                *[
                    Rectangle(height=0.3, width=2, color=BLUE_B, fill_opacity=0.2)
                    for _ in range(10)
                ]
            )
            .arrange(DOWN, buff=0.1)
            .shift(LEFT * 2)
        )
        w_label = MathTex("W", font_size=36).next_to(rows_w, UP)

        # Vector x (Flattened Image, placed next to W)
        vector_x = (
            VGroup(
                *[
                    Square(
                        side_length=0.2,
                        fill_opacity=1,
                        fill_color=interpolate_color(BLUE_E, RED_E, np.random.rand()),
                    )
                    for _ in range(10)
                ]
            )
            .arrange(DOWN, buff=0.05)
            .next_to(rows_w, RIGHT, buff=0.5)
        )
        x_label = MathTex("x", font_size=36).next_to(vector_x, UP)

        # Equal Sign
        eq_sign = MathTex("=", font_size=36).next_to(vector_x, RIGHT, buff=0.5)

        # Scores s (Positioned relative to the equal sign to avoid huge gaps)
        scores = (
            VGroup(*[Rectangle(height=0.3, width=0.8, color=GRAY) for _ in range(10)])
            .arrange(DOWN, buff=0.1)
            .next_to(eq_sign, RIGHT, buff=0.5)
        )
        s_label = MathTex("s", font_size=36).next_to(scores, UP)

        class_names = [
            "Cat",
            "Dog",
            "Ship",
            "Truck",
            "Bird",
            "Frog",
            "Horse",
            "Deer",
            "Plane",
            "Auto",
        ]
        score_labels = VGroup(*[Text(n, font_size=18) for n in class_names])
        for i, label in enumerate(score_labels):
            label.move_to(scores[i].get_center() + RIGHT * 0.9, aligned_edge=LEFT)

        # Animation Sequence 1
        self.play(FadeIn(rows_w, w_label, vector_x, x_label))
        self.play(
            Write(eq_sign),
            FadeIn(scores, s_label, score_labels),
        )

        # Calculation Animation (Row 0) - Slowed down
        row_highlight = SurroundingRectangle(rows_w[0], color=YELLOW)
        score_val_0 = DecimalNumber(
            3.2, num_decimal_places=1, font_size=20, color=GREEN
        ).move_to(scores[0])

        self.play(Create(row_highlight), run_time=1.5)
        self.wait(0.5)
        self.play(Write(score_val_0), scores[0].animate.set_color(GREEN), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(row_highlight))

        # Fill remaining scores
        scores_num = np.zeros(10)
        scores_num[0] = 3.2
        scores_num[1:10] = np.random.uniform(0, 2, size=9)

        other_vals = VGroup(
            *[
                DecimalNumber(
                    scores_num[i], num_decimal_places=1, font_size=20
                ).move_to(scores[i])
                for i in range(1, 10)
            ]
        )
        self.play(FadeIn(other_vals))
        self.wait(1)

        # --- Part 2: Softmax Transition ---
        # Added eq_sign to left_side so it fades out completely
        left_side = VGroup(vector_x, x_label, rows_w, w_label, title, eq_sign)
        self.play(FadeOut(left_side))

        # Re-layout scores to the far left for pipeline
        score_group = VGroup(s_label, scores, score_labels, other_vals, score_val_0)
        self.play(score_group.animate.to_edge(LEFT, buff=1.2).set_y(0))

        softmax_title = Text(
            "Softmax Transformation", font_size=28, color=GOLD
        ).to_edge(UP)
        softmax_formula = MathTex(
            r"p_j = \frac{e^{s_j}}{\sum e^{s_k}}", font_size=40, color=GOLD
        ).move_to(UP * 2 + RIGHT * 2)
        self.play(Write(softmax_title), Write(softmax_formula))

        # Calculation Box (Right)
        calc_box = (
            RoundedRectangle(height=2.5, width=4, color=GRAY, fill_opacity=0.1)
            .to_edge(RIGHT, buff=0.5)
            .set_y(0)
        )
        num_c = MathTex(r"e^{3.2} \approx 24.5", font_size=24, color=GREEN)
        den_c = MathTex(r"\sum e^{s_k} \approx 31.8", font_size=24)
        prob_c = MathTex(r"p_{cat} = 0.77", font_size=32, color=GREEN)
        VGroup(num_c, den_c, prob_c).arrange(DOWN, buff=0.3).move_to(calc_box)

        self.play(Create(calc_box), Write(num_c), Write(den_c))
        self.play(Write(prob_c))

        # --- Part 3: Probability Vector ---
        prob_label = MathTex("p", font_size=36, color=GOLD).move_to(
            s_label.get_center() + RIGHT * 4.5
        )

        raw_scores = scores_num
        exp_scores = np.exp(raw_scores)
        sum_exp = np.sum(exp_scores)
        probabilities = exp_scores / sum_exp

        # Generate bars
        probs = VGroup(
            *[
                Rectangle(
                    height=0.3,
                    width=probabilities[i] * 3.5,
                    color=GREEN if i == 0 else RED,
                    fill_opacity=0.8,
                )
                for i in range(10)
            ]
        ).arrange(DOWN, buff=0.1)

        # Align p-bars to s-bars vertically
        probs.next_to(prob_label, DOWN, buff=0.3).align_to(scores, UP)

        arrow = Arrow(
            scores.get_right() + RIGHT, probs.get_left(), color=GOLD, buff=0.2
        )
        self.play(
            Write(prob_label),
            GrowArrow(arrow),
            LaggedStart(*[Create(p) for p in probs]),
        )

        final_text = Text("Result: CAT", color=GREEN, font_size=32).to_edge(DOWN)
        self.play(probs[0].animate.set_stroke(YELLOW, width=3), Write(final_text))
        self.wait(3)


class RosenbrockFullProcess(ThreeDScene):
    def construct(self):
        # --- SECTION 1: Introduction ---
        intro_text = Text("Gradient Descent", gradient=(BLUE, GREEN)).scale(1.5)
        self.play(Write(intro_text))
        self.wait(2)
        self.play(FadeOut(intro_text))

        # --- SECTION 2: Cost Function Logic ---
        cost_func = MathTex(
            r"J(\theta_0, \theta_1) = (1 - \theta_0)^2 + 100(\theta_1 - \theta_0^2)^2"
        )
        min_text = Text("Goal: Minimize J", color=YELLOW).next_to(cost_func, DOWN)

        self.play(Write(cost_func))
        self.play(FadeIn(min_text, shift=UP))
        self.wait(2)
        self.play(FadeOut(cost_func), FadeOut(min_text))

        # --- SECTION 3: 3D Visualization Setup ---
        def rosenbrock_capped(x, y):
            # Capping Z at 100 to keep the plot readable
            return min((1 - x) ** 2 + 100 * (y - x**2) ** 2, 100)

        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-1, 2, 1],
            z_range=[0, 100, 25],
            x_length=5,
            y_length=5,
            z_length=3,
            axis_config={"include_tip": True},
        )

        surface = Surface(
            lambda u, v: axes.c2p(u, v, rosenbrock_capped(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-0.5, 1.5],
            resolution=(20, 20),
            fill_opacity=0.3,
            fill_color=BLUE_B,
            stroke_color=BLUE_E,
            stroke_width=0.5,
        )

        # --- SECTION 4: Gradient Descent Calculation ---
        curr_pos = np.array([-1.3, 0.2])
        lr = 0.0003
        steps = 12
        path_points = []

        for _ in range(steps):
            x, y = curr_pos
            z = rosenbrock_capped(x, y)
            path_points.append(axes.c2p(x, y, z))

            # Gradients
            dfdx = -2 * (1 - x) - 400 * x * (y - x**2)
            dfdy = 200 * (y - x**2)
            curr_pos = curr_pos - lr * np.array([dfdx, dfdy])

        # --- SECTION 5: The Animation ---
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(axes), Create(surface))
        self.wait(2)

        # Animate discrete jumps
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]

            dot = Sphere(radius=0.1, color=ORANGE).move_to(p1)
            line = Line(p1, p2, color=YELLOW, stroke_width=4)

            self.add(dot)
            self.play(Create(line), run_time=0.2)
            self.wait(1)

        # --- SECTION 6: Outro ---
        self.wait(4)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    # ReLU(x) = max(0, x)
    return np.maximum(0, x)


class ActivationFunctions(Scene):
    def construct(self):
        # --- Titles and Transitions ---
        title = Text("Neural Network Activation Functions").scale(0.9)
        self.play(Write(title))
        self.wait(1.5)
        self.play(FadeOut(title, shift=UP))

        # --- Section 1: Sigmoid Visual ---
        sig_title = Text("1. Sigmoid Function", color=BLUE_B).to_edge(UL).scale(0.8)
        self.play(FadeIn(sig_title, shift=DOWN))

        axes_sig = Axes(
            x_range=[-6, 6, 1],  # Range, step
            y_range=[0, 1.1, 0.5],
            x_length=8,
            y_length=4,
            axis_config={
                "include_tip": True,
                "tip_shape": StealthTip,
            },
            y_axis_config={
                "include_numbers": True,  # Show 0.5 and 1.0
            },
        ).shift(DOWN * 0.5)

        sig_graph = axes_sig.plot(sigmoid, color=BLUE_C, stroke_width=4)
        sig_label = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", color=BLUE_C).next_to(
            sig_graph, UP + RIGHT, buff=-1
        )

        self.play(Create(axes_sig), Create(sig_graph))
        self.play(Write(sig_label))

        # Animate moving dot on Sigmoid
        vt_sig = ValueTracker(-5)
        dot_sig = always_redraw(
            lambda: Dot(color=ORANGE).move_to(
                axes_sig.c2p(vt_sig.get_value(), sigmoid(vt_sig.get_value()))
            )
        )
        self.play(FadeIn(dot_sig, scale=0.5))
        self.play(vt_sig.animate.set_value(5), run_time=3, rate_func=linear)
        self.wait(2)

        # Clear Sigmoid scene
        self.play(
            FadeOut(axes_sig),
            FadeOut(sig_graph),
            FadeOut(sig_label),
            FadeOut(dot_sig),
            FadeOut(sig_title),
        )

        # --- Section 2: ReLU Visual ---
        relu_title = (
            Text("2. Rectified Linear Unit (ReLU)", color=GREEN_B)
            .to_edge(UL)
            .scale(0.8)
        )
        self.play(FadeIn(relu_title, shift=DOWN))

        axes_relu = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 6, 1],
            x_length=8,
            y_length=5,
            axis_config={
                "include_tip": True,
                "tip_shape": StealthTip,
            },
            y_axis_config={
                "include_numbers": True,
            },
        ).shift(DOWN * 0.5)

        relu_graph = axes_relu.plot(relu, color=GREEN_C, stroke_width=5)
        relu_label = MathTex(r"f(x) = \max(0, x)", color=GREEN_C).next_to(
            relu_graph, UP + RIGHT, buff=-1
        )

        self.play(Create(axes_relu), Create(relu_graph))
        self.play(Write(relu_label))

        # Animate moving dot on ReLU
        vt_relu = ValueTracker(-5)
        dot_relu = always_redraw(
            lambda: Dot(color=ORANGE).move_to(
                axes_relu.c2p(vt_relu.get_value(), relu(vt_relu.get_value()))
            )
        )
        self.play(FadeIn(dot_relu, scale=0.5))
        self.play(vt_relu.animate.set_value(5), run_time=3, rate_func=linear)
        self.wait(3)
