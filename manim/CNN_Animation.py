from manim import *
import numpy as np


class Convolution2D(Scene):
    def construct(self):
        # --- Titles and Formulas ---
        title = Tex("2D Convolution Visualization").to_edge(UP)
        formula = (
            MathTex(r"H(x, y) = \sum_{i} \sum_{j} F(x+i, y+j) \cdot G(i, j)")
            .next_to(title, DOWN, buff=0.3)
            .scale(0.8)
        )
        self.add(title, formula)

        # --- Define Data ---
        image_data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 100, 50, 0],
                [0, 100, 200, 100, 0],
                [0, 50, 100, 50, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        kernel_data = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 10.0
        output_shape = (3, 3)

        # --- Create Matrices ---
        image_m = Matrix(image_data).scale(0.6).to_edge(LEFT, buff=1)
        image_label = Tex("Image $F$").next_to(image_m, UP)

        kernel_m = Matrix(kernel_data).scale(0.6).next_to(image_m, RIGHT, buff=1)
        kernel_label = Tex("Kernel $G$").next_to(kernel_m, UP)

        output_m = Matrix(np.zeros(output_shape)).scale(0.6).to_edge(RIGHT, buff=1)
        output_label = Tex("Output $H$").next_to(output_m, UP)

        self.add(
            image_m,
            image_label,
            kernel_m,
            kernel_label,
            output_m.brackets,
            output_label,
        )

        # Helper to get a cell from a Matrix object
        def get_matrix_cell(matrix, row, col):
            # matrix.get_rows() returns a list of VGroups (rows)
            return matrix.get_rows()[row - 1][col - 1]

        # --- Sliding Window Setup ---
        # We want to highlight the 3x3 area. We use the top-left and bottom-right cells.
        ul_cell = get_matrix_cell(image_m, 1, 1)
        dr_cell = get_matrix_cell(image_m, 3, 3)

        sliding_window = SurroundingRectangle(
            VGroup(get_matrix_cell(image_m, 1, 1), get_matrix_cell(image_m, 3, 3)),
            color=ORANGE,
            buff=0.1,
        )

        arrow = Arrow(kernel_m.get_right(), output_m.get_left(), color=YELLOW)
        self.play(Create(sliding_window), GrowArrow(arrow))

        # --- Animation Loop ---
        for r in range(output_shape[0]):
            for c in range(output_shape[1]):
                # Move window to cover the current 3x3 block
                # The block starts at row r+1, col c+1
                current_block = VGroup(
                    get_matrix_cell(image_m, r + 1, c + 1),
                    get_matrix_cell(image_m, r + 3, c + 3),
                )

                # Calculate value (simplified for visual)
                window_data = image_data[r : r + 3, c : c + 3]
                val = np.sum(np.multiply(window_data, kernel_data))

                self.play(
                    sliding_window.animate.move_to(current_block.get_center()),
                    run_time=0.5,
                )

                # Target cell in output matrix
                target_cell = get_matrix_cell(output_m, r + 1, c + 1)
                res_val = (
                    DecimalNumber(val, num_decimal_places=1)
                    .scale(0.5)
                    .move_to(target_cell)
                )

                highlighter = SurroundingRectangle(target_cell, color=RED)

                self.play(Create(highlighter), FadeIn(res_val), run_time=0.4)
                self.play(FadeOut(highlighter), run_time=0.2)

        self.wait(2)


class WhiskerMaxPooling(Scene):
    def construct(self):
        # --- Titles and Setup ---
        title = Text("Max Pooling: Matrix Spatial Reduction", font_size=40).to_edge(UP)
        self.play(Write(title))

        # Define the Input Feature Map (4x4 Matrix)
        # 0.9 represents the strongest whisker activation
        input_data = [
            [0.1, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.3, 0.1],
            [0.0, 0.2, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.1],
        ]

        # Create Input Matrix
        input_matrix = Matrix(input_data).scale(0.7).shift(LEFT * 3)
        matrix_label = Text("4x4 Feature Map (Whiskers)", font_size=24).next_to(
            input_matrix, UP
        )

        self.play(FadeIn(input_matrix), FadeIn(matrix_label))
        self.wait(1)

        # Highlight the key feature (the '0.9' whisker detection)
        # Matrix.get_entries() returns a flat list. Index 5 is Row 2, Col 2.
        strong_feature = input_matrix.get_entries()[5]
        self.play(strong_feature.animate.set_color(RED), run_time=1)

        # --- Prepare Output Matrix ---
        # Initialize a 2x2 Matrix with placeholders
        output_matrix = Matrix([[0.0, 0.0], [0.0, 0.0]]).scale(0.7).shift(RIGHT * 3)
        output_label = Text("2x2 Pooled Output", font_size=24).next_to(
            output_matrix, UP
        )

        # Calculate Window Geometry
        # We find the distance between matrix entries to size the sliding window
        entries = input_matrix.get_entries()
        dx = entries[1].get_x() - entries[0].get_x()
        dy = entries[0].get_y() - entries[4].get_y()

        # Pooling Window (Yellow Rectangle)
        pool_window = Rectangle(
            width=dx * 1.9, height=dy * 1.9, color=YELLOW, stroke_width=4
        )
        # Center the window on the top-left 2x2 quadrant
        first_quad_center = VGroup(
            entries[0], entries[1], entries[4], entries[5]
        ).get_center()
        pool_window.move_to(first_quad_center)

        self.play(
            FadeIn(output_matrix.brackets), FadeIn(output_label), Create(pool_window)
        )
        self.wait(1)

        # --- Max Pooling Step-by-Step Logic ---
        # quadrants maps input indices to output indices and defines window movement
        quadrants = [
            {"input_idx": [0, 1, 4, 5], "out_idx": 0, "shift_dir": ORIGIN},
            {"input_idx": [2, 3, 6, 7], "out_idx": 1, "shift_dir": RIGHT * dx * 2},
            {
                "input_idx": [8, 9, 12, 13],
                "out_idx": 2,
                "shift_dir": DOWN * dy * 2 + LEFT * dx * 2,
            },
            {"input_idx": [10, 11, 14, 15], "out_idx": 3, "shift_dir": RIGHT * dx * 2},
        ]

        for i, quad in enumerate(quadrants):
            # 1. Slide the window
            if i > 0:
                self.play(pool_window.animate.shift(quad["shift_dir"]), run_time=0.6)

            # 2. Identify the active quadrant
            active_entries = VGroup(*[entries[idx] for idx in quad["input_idx"]])
            self.play(Indicate(active_entries, color=YELLOW, scale_factor=1.1))

            # 3. Extract the maximum value from the data
            current_vals = [input_data[idx // 4][idx % 4] for idx in quad["input_idx"]]
            max_val = max(current_vals)

            # 4. Animate the Max value flying to the output matrix
            target_entry = output_matrix.get_entries()[quad["out_idx"]]
            moving_val = DecimalNumber(max_val, num_decimal_places=1).scale(0.7)
            moving_val.move_to(active_entries.get_center())

            self.play(
                moving_val.animate.move_to(target_entry.get_center()), run_time=0.8
            )
            # Ensure the output entry is highlighted
            self.add(moving_val.set_color(YELLOW))
            self.wait(0.5)

        # --- Conclusion / Summary ---
        self.play(FadeOut(pool_window))

        summary = (
            VGroup(
                Text("Max Pooling Principles:", font_size=25, color=BLUE),
                Text("1. Downsampling: 4x4 matrix shrinks to 2x2.", font_size=20),
                Text("2. Invariance: The large signal survived.", font_size=20),
                Text(
                    "The model stays robust to minor feature shifts.",
                    font_size=20,
                    color=GRAY,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(DOWN)
        )

        self.play(Write(summary))

        self.wait(3)
